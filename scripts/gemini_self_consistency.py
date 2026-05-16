"""Self-consistency scorer: average N Gemini samples per example.

Strategy:
  - Sample at temperature > 0 N times per example.
  - Each sample returns an integer 1-5.
  - Final score = mean(samples), a continuous value in [1, 5].
  - This gives Gemini a "calibrated continuous" output that better matches
    AmbiStory's continuous human-average labels.

Reuses the better prompt from gemini_score_v2.PROMPT_BETTER.

Usage:
    GEMINI_API_KEY=... python gemini_self_consistency.py \\
        --input data/dev.json \\
        --out results/gemini_sc5_dev.json \\
        --samples 5 --temperature 0.7 --concurrency 32
"""

import argparse
import asyncio
import functools
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

print = functools.partial(print, flush=True)

MODEL = "gemini-2.5-pro"

PROMPT_BETTER = """You will rate how plausible a candidate meaning of a homonym is, given a surrounding short narrative.

Use this 1-5 scale:
- 1: The meaning is incompatible with the context; a reader would never consider it here.
- 2: Theoretically possible but heavily disfavored - another meaning fits the context much better.
- 3: The narrative is genuinely ambiguous. This meaning and at least one other are about equally plausible. Many items rightly belong here - do not avoid 3.
- 4: This is the most natural reading, but other meanings remain conceivable.
- 5: This is the ONLY coherent reading; alternative meanings would make the narrative inconsistent or absurd.

Calibration notes:
- Use the full 1-5 range. Do not default to 4 or 5 just because the meaning fits.
- The homonym may appear in inflected form (e.g., past tense, plural) - count any morphological form.
- The ending often disambiguates - read it carefully before deciding.
- Reserve 5 for cases where the context actively rules out alternative meanings.

Narrative: {precontext} **{sentence}** {ending}

Candidate: the word "{word}" means "{word_sense}" (example usage: "{example_sentence}")

Return ONLY a single integer from 1, 2, 3, 4, or 5. Do not include explanation or any other text."""


def render_prompt(row):
    ending = row["ending"] if isinstance(row["ending"], str) and row["ending"] else ""
    return PROMPT_BETTER.format(
        precontext=row["precontext"], sentence=row["sentence"], ending=ending,
        word=row["homonym"], word_sense=row["judged_meaning"], example_sentence=row["example_sentence"],
    )


def parse_int_score(text):
    if not text:
        return None
    t = text.strip()
    if t in "12345":
        return int(t)
    m = re.search(r"\b([1-5])\b", t)
    return int(m.group(1)) if m else None


async def one_sample(client, semaphore, prompt, thinking_budget, temperature, per_call_timeout=90.0, max_retries=5):
    from google.genai import types
    cfg = {"temperature": temperature, "max_output_tokens": max((thinking_budget or 0), 0) + 64}
    if thinking_budget is not None:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    config = types.GenerateContentConfig(**cfg)
    async with semaphore:
        for attempt in range(max_retries):
            try:
                resp = await asyncio.wait_for(
                    client.aio.models.generate_content(model=MODEL, contents=prompt, config=config),
                    timeout=per_call_timeout,
                )
                v = parse_int_score(resp.text)
                if v is None:
                    raise ValueError(f"parse failed: {resp.text!r}")
                return v
            except asyncio.TimeoutError:
                if attempt == max_retries - 1: return None
                await asyncio.sleep(1.0)
            except Exception as e:
                msg = str(e)
                if attempt == max_retries - 1: return None
                wait = (2.0 ** attempt) + (0.5 * attempt)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                    wait = max(wait, 10.0 + 5.0 * attempt)
                await asyncio.sleep(wait)
        return None


async def score_with_self_consistency(client, semaphore, key, prompt, samples, thinking_budget, temperature):
    coros = [one_sample(client, semaphore, prompt, thinking_budget, temperature) for _ in range(samples)]
    results = await asyncio.gather(*coros)
    valid = [v for v in results if v is not None]
    if not valid:
        return key, None, results
    return key, float(np.mean(valid)), results


async def run_async(json_path, out_path, samples, temperature, concurrency, thinking_budget, n_samples=None):
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    df = pd.read_json(json_path).T
    if n_samples is not None:
        df = df.head(n_samples)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    preds = {}
    raw = {}
    if Path(out_path).exists():
        cache = json.load(open(out_path))
        preds = {k: v.get("mean") for k, v in cache.items()}
        raw = {k: v.get("samples", []) for k, v in cache.items()}
        print(f"resumed {sum(v is not None for v in preds.values())} cached preds")

    todo = [(str(sid), render_prompt(row)) for sid, row in df.iterrows() if preds.get(str(sid)) is None]
    print(f"scoring {len(todo)} examples  samples={samples}  temp={temperature}  thinking={thinking_budget}  concurrency={concurrency}")
    if not todo:
        return preds, raw

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [score_with_self_consistency(client, semaphore, key, prompt, samples, thinking_budget, temperature) for key, prompt in todo]
    start = time.time()
    done, last_save, fails = 0, 0, 0
    for fut in asyncio.as_completed(tasks):
        key, mean_score, sample_list = await fut
        preds[key] = mean_score
        raw[key] = sample_list
        done += 1
        if mean_score is None:
            fails += 1
        if done - last_save >= 20:
            json.dump({k: {"mean": preds[k], "samples": raw[k]} for k in preds}, open(out_path, "w"), indent=2)
            last_save = done
            rate = done / max(time.time() - start, 1e-3)
            eta = (len(todo) - done) / max(rate, 1e-3)
            print(f"  progress {done}/{len(todo)}  rate={rate:.2f}/s  eta={eta/60:.1f}m  fails={fails}")
    json.dump({k: {"mean": preds[k], "samples": raw[k]} for k in preds}, open(out_path, "w"), indent=2)
    print(f"\ndone in {(time.time()-start)/60:.1f}m  ({done} new, {fails} failed)")
    return preds, raw


def report_metrics(json_path, out_path):
    df = pd.read_json(json_path).T
    cache = json.load(open(out_path))
    preds = {k: v.get("mean") for k, v in cache.items()}
    keep = [(str(k), preds.get(str(k))) for k in df.index if preds.get(str(k)) is not None]
    if not keep:
        return None
    ids, y_pred = zip(*keep)
    y_pred = np.array(y_pred, dtype=float)
    sub = df.set_index(df.index.astype(str)).loc[list(ids)]
    if sub["average"].dtype == object:
        if sub["average"].apply(lambda v: not isinstance(v, (int, float))).any():
            print(f"\nNo gold labels in {Path(json_path).name}.")
            return None
    y_true = sub["average"].to_numpy(dtype=float)
    rho, _ = spearmanr(y_pred, y_true)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    sigma = sub["stdev"].to_numpy(dtype=float)
    acc_sd = float(np.mean(np.abs(y_pred - y_true) <= np.maximum(sigma, 1.0)))
    m = {"spearman": float(rho), "mae": mae, "rmse": rmse, "acc_within_sd": acc_sd, "n": int(len(y_pred))}
    print(f"\n== self-consistency on {Path(json_path).name} ==")
    print(f"  N={m['n']}")
    print(f"  Spearman:      {m['spearman']:.4f}")
    print(f"  MAE:           {m['mae']:.4f}")
    print(f"  RMSE:          {m['rmse']:.4f}")
    print(f"  Acc-within-SD: {m['acc_within_sd']:.4f}")
    json.dump(m, open(out_path.replace(".json", "_metrics.json"), "w"), indent=2)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--thinking-budget", type=int, default=128)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--n", type=int, default=None)
    args = p.parse_args()
    asyncio.run(run_async(args.input, args.out, args.samples, args.temperature, args.concurrency, args.thinking_budget, n_samples=args.n))
    report_metrics(args.input, args.out)


if __name__ == "__main__":
    main()

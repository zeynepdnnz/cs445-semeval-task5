"""Parallel async Gemini 2.5 Pro scorer for AmbiStory.

Same prompt as the GPT-4o-mini baseline in CS445_Baselines.ipynb cell 16
(integer mode) or a continuous-score variant (for distillation soft labels).

Concurrency, retries with exponential backoff, resumable cache.

Usage:
    GEMINI_API_KEY=... python gemini_score_async.py \\
        --input data/test.json \\
        --out results/gemini_test_int.json \\
        --prompt-mode integer \\
        --concurrency 16

    # Smoke test:
    python gemini_score_async.py --input data/test.json \\
        --out results/smoke.json --n 10
"""

import argparse
import asyncio
import functools
import json
import os
import re
import sys
import time

# Unbuffer stdout so progress prints stream when output is captured to a file.
print = functools.partial(print, flush=True)
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MODEL = "gemini-2.5-pro"

PROMPT_INTEGER = """You will see a short text in which one sentence is marked with "**". That sentence contains a word that can typically take on multiple different meanings, depending on the context. One of those meanings is given to you.

**Your task is simple: Annotate how plausible a meaning of a word is in the context of the short text using one of five scores:**

* **1**: The displayed meaning is not plausible at all given the context.
* **2**: The displayed meaning is theoretically conceivable, but less plausible than other meanings.
* **3**: The displayed meaning represents one of multiple, similarly plausible interpretations.
* **4**: The displayed meaning represents the most plausible interpretation; other meanings may still be conceivable.
* **5**: The displayed meaning is the only plausible meaning given the context.

There will be times where there is no objectively correct answer. Whatever the case, always look at all of the sentences and carefully think about how plausible each meaning would be.

Now take a look at the following text: {precontext} **{sentence}** {ending}

In this context, how plausible is it that the meaning of the word "{word}" is "{word_sense}"?

Return only the numbered score (1, 2, 3, 4 or 5). Do not return anything else!"""

PROMPT_CONTINUOUS = """You will see a short text in which one sentence is marked with "**". That sentence contains a word that can typically take on multiple different meanings, depending on the context. One of those meanings is given to you.

Rate how plausible the given meaning is in the context of the short text on a continuous scale from 1.0 to 5.0:

* 1.0: The meaning is not plausible at all given the context.
* 2.0: The meaning is theoretically conceivable, but less plausible than others.
* 3.0: The meaning is one of multiple, similarly plausible interpretations.
* 4.0: The meaning is the most plausible interpretation; others may still be conceivable.
* 5.0: The meaning is the only plausible interpretation given the context.

You may (and should) use values between integers (e.g. 2.4, 3.7) to express finer judgments. Many examples are genuinely ambiguous - do not collapse to a round integer when a fractional rating is more honest.

Text: {precontext} **{sentence}** {ending}

In this context, how plausible is the meaning of the word "{word}" being "{word_sense}"?

Return only the number (e.g. 3.4). Do not return anything else."""


def build_prompt(row, prompt_mode):
    template = PROMPT_INTEGER if prompt_mode == "integer" else PROMPT_CONTINUOUS
    ending = row["ending"] if isinstance(row["ending"], str) and row["ending"] else ""
    return template.format(
        precontext=row["precontext"],
        sentence=row["sentence"],
        ending=ending,
        word=row["homonym"],
        word_sense=row["judged_meaning"],
    )


def parse_score(text, prompt_mode):
    if not text:
        return None
    text = text.strip()
    if prompt_mode == "integer":
        if text in "12345":
            return float(text)
        m = re.search(r"\b([1-5])\b", text)
        return float(m.group(1)) if m else None
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    return max(1.0, min(5.0, float(m.group(1))))


async def score_one(client, semaphore, key, prompt, prompt_mode, thinking_budget, max_retries=5, per_call_timeout=90.0):
    from google.genai import types
    cfg_kwargs = {"temperature": 0.0, "max_output_tokens": (thinking_budget or 0) + 64}
    if thinking_budget is not None:
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    config = types.GenerateContentConfig(**cfg_kwargs)
    async with semaphore:
        for attempt in range(max_retries):
            try:
                resp = await asyncio.wait_for(
                    client.aio.models.generate_content(model=MODEL, contents=prompt, config=config),
                    timeout=per_call_timeout,
                )
                score = parse_score(resp.text, prompt_mode)
                if score is None:
                    raise ValueError(f"parse failed (finish={getattr(resp.candidates[0], 'finish_reason', '?')}, text={resp.text!r})")
                return key, score, None
            except asyncio.TimeoutError:
                if attempt == max_retries - 1:
                    return key, None, f"timeout after {per_call_timeout}s"
                await asyncio.sleep(1.0)
            except Exception as e:
                msg = str(e)
                if attempt == max_retries - 1:
                    return key, None, msg
                wait = (2.0 ** attempt) + (0.5 * attempt)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                    wait = max(wait, 10.0 + 5.0 * attempt)
                await asyncio.sleep(wait)
        return key, None, "exhausted retries"


async def run_async(json_path, out_path, prompt_mode, concurrency, n_samples=None, save_every=25):
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
    if Path(out_path).exists():
        preds = json.load(open(out_path))
        n_cached = sum(v is not None for v in preds.values())
        print(f"resumed {n_cached} cached preds from {out_path}")

    todo = [(str(sid), build_prompt(row, prompt_mode)) for sid, row in df.iterrows()
            if preds.get(str(sid)) is None]
    print(f"scoring {len(todo)} new examples (concurrency={concurrency}, mode={prompt_mode})")

    if not todo:
        print("nothing to do")
        return preds

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [score_one(client, semaphore, key, prompt, prompt_mode, thinking_budget=128) for key, prompt in todo]

    start = time.time()
    done_count = 0
    last_save = 0
    fails = 0
    for fut in asyncio.as_completed(tasks):
        key, score, err = await fut
        preds[key] = score
        done_count += 1
        if score is None:
            fails += 1
            if fails <= 3:
                print(f"  fail {key}: {err}")
        if done_count - last_save >= save_every:
            json.dump(preds, open(out_path, "w"), indent=2)
            last_save = done_count
            rate = done_count / max(time.time() - start, 1e-3)
            eta_s = (len(todo) - done_count) / max(rate, 1e-3)
            print(f"  progress {done_count}/{len(todo)}  rate={rate:.2f}/s  eta={eta_s/60:.1f}m  fails={fails}")
    json.dump(preds, open(out_path, "w"), indent=2)
    elapsed = time.time() - start
    print(f"\ndone in {elapsed/60:.1f}m  ({done_count} new, {fails} failed)  -> {out_path}")
    return preds


def report_metrics(json_path, out_path):
    df = pd.read_json(json_path).T
    preds = json.load(open(out_path))
    keep = [(str(k), preds.get(str(k))) for k in df.index if preds.get(str(k)) is not None]
    if not keep:
        print("no preds to score")
        return None
    ids, y_pred = zip(*keep)
    y_pred = np.array(y_pred, dtype=float)
    sub = df.loc[[int(k) if str(k).isdigit() else k for k in ids]] if not isinstance(df.index[0], str) else df.loc[list(ids)]
    # SemEval-2026 hides test labels with '(???)' placeholders. Skip metrics in that case.
    if sub["average"].dtype == object:
        non_numeric = sub["average"].apply(lambda v: not isinstance(v, (int, float))).any()
        if non_numeric:
            print(f"\nNo gold labels in {Path(json_path).name} (placeholder values present). "
                  f"Predictions saved to {out_path} for submission only.")
            return None
    y_true = sub["average"].to_numpy(dtype=float)
    rho, _ = spearmanr(y_pred, y_true)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    acc_sd = None
    if "stdev" in sub.columns:
        sigma = sub["stdev"].to_numpy(dtype=float)
        acc_sd = float(np.mean(np.abs(y_pred - y_true) <= np.maximum(sigma, 1.0)))
    m = {"spearman": float(rho), "mae": mae, "rmse": rmse, "acc_within_sd": acc_sd, "n": int(len(y_pred))}
    print(f"\n== Gemini {MODEL} on {Path(json_path).name} ==")
    print(f"  N={m['n']}")
    print(f"  Spearman:      {m['spearman']:.4f}")
    print(f"  MAE:           {m['mae']:.4f}")
    print(f"  RMSE:          {m['rmse']:.4f}")
    if acc_sd is not None:
        print(f"  Acc-within-SD: {m['acc_within_sd']:.4f}")
    mpath = out_path.replace(".json", "_metrics.json")
    json.dump(m, open(mpath, "w"), indent=2)
    print(f"metrics -> {mpath}")
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prompt-mode", choices=["integer", "continuous"], default="integer")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--n", type=int, default=None, help="smoke-test N rows")
    p.add_argument("--no-metrics", action="store_true")
    args = p.parse_args()

    asyncio.run(run_async(args.input, args.out, args.prompt_mode, args.concurrency, n_samples=args.n))

    if not args.no_metrics:
        report_metrics(args.input, args.out)


if __name__ == "__main__":
    main()

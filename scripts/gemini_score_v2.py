"""Gemini 2.5 Pro scorer with multiple prompt + thinking variants.

Prompt styles:
  paper           - same as CS445_Baselines.ipynb cell 16 (GPT-4o-mini protocol).
  paper-oneshot   - paper prompt with one demonstration (rating 3, teaches calibration).
  better          - rewritten prompt with explicit rubric + calibration notes.
  better-oneshot  - better prompt + the same demonstration.

Thinking modes (Gemini 2.5 Pro):
  --thinking-budget 128   - default, model reasons internally before answering.
  --thinking-budget 0     - thinking disabled where supported. (Pro may still
                            allocate some tokens; we set output budget accordingly.)

Concurrency, retries, resumable cache, per-call timeout.

Usage:
    GEMINI_API_KEY=... python gemini_score_v2.py \\
        --input data/dev.json \\
        --out results/gemini_dev_paper_nothink.json \\
        --prompt-style paper \\
        --thinking-budget 0 \\
        --concurrency 32
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

PROMPT_PAPER = """You will see a short text in which one sentence is marked with "**". That sentence contains a word that can typically take on multiple different meanings, depending on the context. One of those meanings is given to you.

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


# One demonstration drawn from train (sample_id=402, avg=3.00, stdev=0.00).
# Teaches the model that ambiguous loan/curiosity overlap should be a 3,
# not a confident 4 or 5. The model in our zero-shot run over-predicts toward
# the upper end; this calibration anchor pulls it back toward the human mean.
ONE_SHOT_DEMO = """Here is one rated example to calibrate your judgments:

Narrative: John opened the envelope with trembling hands. He had applied for a loan last week to help with his small business. **There was a large interest shown.**

Candidate: the word "interest" means "a fixed charge for borrowing money; usually a percentage of the amount borrowed" (example usage: "How much interest do you owe?")

Rating: 3

(Reasoning: although the precontext mentions a loan, the surface phrasing "interest was shown" much more naturally evokes the "curiosity / attention" sense. Both readings remain available, so the meaning is one of multiple similarly plausible interpretations - hence 3, not 4 or 5.)

Now rate the next item using the same scale:

"""


def render_prompt(row, style):
    ending = row["ending"] if isinstance(row["ending"], str) and row["ending"] else ""
    if style.startswith("better"):
        body = PROMPT_BETTER.format(
            precontext=row["precontext"], sentence=row["sentence"], ending=ending,
            word=row["homonym"], word_sense=row["judged_meaning"], example_sentence=row["example_sentence"],
        )
    else:
        body = PROMPT_PAPER.format(
            precontext=row["precontext"], sentence=row["sentence"], ending=ending,
            word=row["homonym"], word_sense=row["judged_meaning"],
        )
    if style.endswith("oneshot"):
        return ONE_SHOT_DEMO + body
    return body


def parse_int_score(text):
    if not text:
        return None
    t = text.strip()
    if t in "12345":
        return float(t)
    m = re.search(r"\b([1-5])\b", t)
    return float(m.group(1)) if m else None


async def score_one(client, semaphore, key, prompt, thinking_budget, model, per_call_timeout=90.0, max_retries=5):
    from google.genai import types
    cfg = {"temperature": 0.0, "max_output_tokens": max((thinking_budget or 0), 0) + 64}
    if thinking_budget is not None:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    config = types.GenerateContentConfig(**cfg)

    async with semaphore:
        for attempt in range(max_retries):
            try:
                resp = await asyncio.wait_for(
                    client.aio.models.generate_content(model=model, contents=prompt, config=config),
                    timeout=per_call_timeout,
                )
                score = parse_int_score(resp.text)
                if score is None:
                    finish = getattr(resp.candidates[0], "finish_reason", "?") if getattr(resp, "candidates", None) else "?"
                    raise ValueError(f"parse failed (finish={finish}, text={resp.text!r})")
                return key, score, None
            except asyncio.TimeoutError:
                if attempt == max_retries - 1:
                    return key, None, f"timeout {per_call_timeout}s"
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


async def run_async(json_path, out_path, style, concurrency, thinking_budget, model, n_samples=None, save_every=25):
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
        print(f"resumed {sum(v is not None for v in preds.values())} cached preds from {out_path}")

    todo = [(str(sid), render_prompt(row, style)) for sid, row in df.iterrows() if preds.get(str(sid)) is None]
    print(f"scoring {len(todo)} examples  model={model}  style={style}  thinking={thinking_budget}  concurrency={concurrency}")
    if not todo:
        return preds

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [score_one(client, semaphore, key, prompt, thinking_budget, model) for key, prompt in todo]
    start = time.time()
    done, last_save, fails = 0, 0, 0
    for fut in asyncio.as_completed(tasks):
        key, score, err = await fut
        preds[key] = score
        done += 1
        if score is None:
            fails += 1
            if fails <= 5:
                print(f"  fail {key}: {err}")
        if done - last_save >= save_every:
            json.dump(preds, open(out_path, "w"), indent=2)
            last_save = done
            rate = done / max(time.time() - start, 1e-3)
            eta = (len(todo) - done) / max(rate, 1e-3)
            print(f"  progress {done}/{len(todo)}  rate={rate:.2f}/s  eta={eta/60:.1f}m  fails={fails}")
    json.dump(preds, open(out_path, "w"), indent=2)
    print(f"\ndone in {(time.time()-start)/60:.1f}m  ({done} new, {fails} failed)  -> {out_path}")
    return preds


def report_metrics(json_path, out_path):
    df = pd.read_json(json_path).T
    preds = json.load(open(out_path))
    keep = [(str(k), preds.get(str(k))) for k in df.index if preds.get(str(k)) is not None]
    if not keep:
        print("no preds")
        return None
    ids, y_pred = zip(*keep)
    y_pred = np.array(y_pred, dtype=float)
    sub = df.set_index(df.index.astype(str)).loc[list(ids)]
    if sub["average"].dtype == object:
        if sub["average"].apply(lambda v: not isinstance(v, (int, float))).any():
            print(f"\nNo gold labels in {Path(json_path).name} (placeholders).")
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
    print(f"\n== {MODEL} on {Path(json_path).name} ==")
    print(f"  N={m['n']}")
    print(f"  Spearman:      {m['spearman']:.4f}")
    print(f"  MAE:           {m['mae']:.4f}")
    print(f"  RMSE:          {m['rmse']:.4f}")
    if acc_sd is not None:
        print(f"  Acc-within-SD: {m['acc_within_sd']:.4f}")
    json.dump(m, open(out_path.replace(".json", "_metrics.json"), "w"), indent=2)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prompt-style", choices=["paper", "paper-oneshot", "better", "better-oneshot"], default="paper")
    p.add_argument("--thinking-budget", type=int, default=128, help="thinking budget; 0 only works for Flash")
    p.add_argument("--model", default=MODEL, help="e.g. gemini-2.5-pro (default) or gemini-2.5-flash")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--n", type=int, default=None)
    args = p.parse_args()
    asyncio.run(run_async(args.input, args.out, args.prompt_style, args.concurrency, args.thinking_budget, args.model, n_samples=args.n))
    report_metrics(args.input, args.out)


if __name__ == "__main__":
    main()

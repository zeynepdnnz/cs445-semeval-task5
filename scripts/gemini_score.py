"""Score AmbiStory examples with Gemini 2.5 Pro.

Two purposes:
  1. Zero-shot baseline on test/dev (reports Spearman / MAE / RMSE / acc_within_sd).
  2. Soft-label generation on train for distillation into DeBERTa.

Colab usage:
    !pip install -q google-genai
    from google.colab import userdata
    import os
    os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')
    from gemini_score import score_split

    # Zero-shot baseline on test (integer prompt matches the paper protocol):
    score_split('/content/test.json',
                '/content/drive/MyDrive/gemini_test_int.json',
                prompt_mode='integer')

    # Soft labels on train for distillation (continuous 1.0-5.0):
    score_split('/content/train.json',
                '/content/drive/MyDrive/gemini_train_soft.json',
                prompt_mode='continuous')

CLI usage:
    python gemini_score.py --input test.json --out gemini_test.json \\
        --prompt-mode integer
"""

import argparse
import json
import os
import re
import time
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

You may (and should) use values between integers (e.g. 2.4, 3.7) to express finer judgments. Many examples are genuinely ambiguous — do not collapse to a round integer when a fractional rating is more honest.

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
    val = float(m.group(1))
    return max(1.0, min(5.0, val))


def _compute_metrics(df, preds):
    keep = [(k, v) for k, v in preds.items() if v is not None and k in df.index.astype(str)]
    if not keep:
        return None
    ids, y_pred = zip(*keep)
    y_pred = np.array(y_pred, dtype=float)
    sub = df.set_index(df.index.astype(str)).loc[list(ids)]
    y_true = sub["average"].to_numpy(dtype=float)
    rho, _ = spearmanr(y_pred, y_true)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    acc_sd = None
    if "stdev" in sub.columns:
        sigma = sub["stdev"].to_numpy(dtype=float)
        acc_sd = float(np.mean(np.abs(y_pred - y_true) <= np.maximum(sigma, 1.0)))
    return {"spearman": float(rho), "mae": mae, "rmse": rmse, "acc_within_sd": acc_sd, "n": int(len(y_pred))}


def score_split(
    json_path,
    out_path,
    prompt_mode="continuous",
    model=MODEL,
    temperature=0.0,
    resume=True,
    n_samples=None,
    save_every=50,
    sleep_between=0.0,
):
    """Score one AmbiStory split with Gemini.

    Args:
        json_path: train.json / dev.json / test.json
        out_path: JSON file to write {sample_id: score}. Re-running resumes from it.
        prompt_mode: "integer" (replicates the paper's GPT-4o-mini protocol) or
                     "continuous" (1.0-5.0, recommended for distillation soft labels).
        temperature: 0.0 = deterministic. Use 0.7 + multiple runs if you want
                     an ensemble-of-judgments soft label.
        n_samples: smoke-test on first N rows.
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in env (Colab: userdata.get('GEMINI_API_KEY'))")

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(temperature=temperature, max_output_tokens=16)

    df = pd.read_json(json_path).T
    if n_samples is not None:
        df = df.head(n_samples)
    print(f"Scoring {len(df)} examples from {json_path} with {model} ({prompt_mode})")

    preds = {}
    if resume and Path(out_path).exists():
        preds = json.load(open(out_path))
        print(f"  resumed {sum(v is not None for v in preds.values())} cached preds")

    fails = 0
    for sid, row in df.iterrows():
        key = str(sid)
        if preds.get(key) is not None:
            continue
        prompt = build_prompt(row, prompt_mode)
        score = None
        last_err = None
        for attempt in range(4):
            try:
                resp = client.models.generate_content(model=model, contents=prompt, config=config)
                score = parse_score(resp.text, prompt_mode)
                if score is None:
                    raise ValueError(f"parse failed: {resp.text!r}")
                break
            except Exception as e:
                last_err = e
                if attempt < 3:
                    time.sleep(2.0 * (2 ** attempt))
        if score is None:
            print(f"  {key} failed: {last_err}")
            fails += 1
        preds[key] = score
        if sleep_between:
            time.sleep(sleep_between)
        if len(preds) % save_every == 0:
            json.dump(preds, open(out_path, "w"), indent=2)
            print(f"  cached {len(preds)}/{len(df)}")

    json.dump(preds, open(out_path, "w"), indent=2)
    print(f"Saved -> {out_path}  (failures={fails})")

    if "average" in df.columns:
        m = _compute_metrics(df, preds)
        if m is not None:
            print(f"\nGemini {model} on {Path(json_path).name} ({prompt_mode}):")
            print(f"  N={m['n']}")
            print(f"  Spearman:      {m['spearman']:.4f}")
            print(f"  MAE:           {m['mae']:.4f}")
            print(f"  RMSE:          {m['rmse']:.4f}")
            if m["acc_within_sd"] is not None:
                print(f"  Acc-within-SD: {m['acc_within_sd']:.4f}")
            metrics_path = out_path.replace(".json", "_metrics.json")
            json.dump(m, open(metrics_path, "w"), indent=2)
            print(f"  metrics -> {metrics_path}")
    return preds


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="train.json / dev.json / test.json")
    p.add_argument("--out", required=True, help="cache JSON for predictions")
    p.add_argument("--prompt-mode", choices=["integer", "continuous"], default="continuous")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--model", default=MODEL)
    args = p.parse_args()
    score_split(
        args.input,
        args.out,
        prompt_mode=args.prompt_mode,
        model=args.model,
        temperature=args.temperature,
        n_samples=args.n,
    )

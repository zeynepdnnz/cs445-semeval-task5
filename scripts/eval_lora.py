"""Standalone evaluator for a saved LoRA adapter on AmbiStory.

Loads the base model + adapter, runs self-consistency generation on dev + test,
computes metrics, saves predictions.

Usage:
    python eval_lora.py \\
        --base-model Qwen/Qwen3-8B \\
        --adapter-dir /home/ubuntu/cs445/lora_out/qwen3_8b \\
        --out-dir /home/ubuntu/cs445/lora_out/qwen3_8b \\
        --samples 5 --temperature 0.7
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

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
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    text = text.strip()
    if not text:
        return None
    nums = re.findall(r"\b([1-5])\b", text)
    if nums:
        return int(nums[-1])
    if text[0] in "12345":
        return int(text[0])
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-path", default="/home/ubuntu/cs445/data/train.json")
    p.add_argument("--dev-path", default="/home/ubuntu/cs445/data/dev.json")
    p.add_argument("--test-path", default="/home/ubuntu/cs445/data/test_labeled.json")
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=4)
    p.add_argument("--no-shutdown", action="store_true")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base {args.base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only models need left-padding for batched generation
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="cuda")
    print(f"Loading adapter {args.adapter_dir} ...")
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    @torch.inference_mode()
    def score_split(df, tag):
        prompts = [render_prompt(row) for _, row in df.iterrows()]
        chats = []
        for p_ in prompts:
            msgs = [{"role": "user", "content": p_}]
            try:
                chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            chats.append(chat)
        preds = []
        t0 = time.time()
        for i in range(0, len(chats), args.batch_size):
            batch = chats[i:i + args.batch_size]
            toks = tokenizer(batch, padding=True, truncation=True, max_length=1024, return_tensors="pt").to("cuda")
            out = model.generate(
                **toks,
                do_sample=True, temperature=args.temperature, top_p=0.95,
                max_new_tokens=args.max_tokens, num_return_sequences=args.samples,
                pad_token_id=tokenizer.pad_token_id,
            )
            out = out[:, toks["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(out, skip_special_tokens=True)
            for j in range(len(batch)):
                sample_texts = texts[j * args.samples: (j + 1) * args.samples]
                ints = [v for v in (parse_int_score(t) for t in sample_texts) if v is not None]
                preds.append(float(np.mean(ints)) if ints else None)
            if (i // args.batch_size) % 5 == 0:
                rate = (i + len(batch)) / max(time.time() - t0, 1e-3)
                eta = (len(chats) - i - len(batch)) / max(rate, 1e-3)
                print(f"  [{tag}] {i + len(batch)}/{len(chats)}  rate={rate:.2f}/s eta={eta/60:.1f}m")
        return preds

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out = {}
    for tag, path in [("dev", args.dev_path), ("test", args.test_path)]:
        df = pd.read_json(path).T
        df.index = df.index.astype(str)
        # Robust label parsing — handle stored-as-object numeric.
        try:
            y_true = df["average"].astype(float).to_numpy()
        except (ValueError, TypeError):
            print(f"Skip {tag} metrics (placeholder labels)")
            preds = score_split(df, tag)
            cache = {str(k): {"mean": v} for k, v in zip(df.index, preds)}
            json.dump(cache, open(f"{args.out_dir}/preds_{tag}.json", "w"), indent=2)
            continue
        sigma = df["stdev"].astype(float).to_numpy()
        preds = score_split(df, tag)
        cache = {str(k): {"mean": v} for k, v in zip(df.index, preds)}
        json.dump(cache, open(f"{args.out_dir}/preds_{tag}.json", "w"), indent=2)
        valid = [i for i, v in enumerate(preds) if v is not None]
        y_pred = np.array([preds[i] for i in valid])
        y_true_v = y_true[valid]
        sig_v = sigma[valid]
        rho, _ = spearmanr(y_pred, y_true_v)
        mae = float(np.mean(np.abs(y_pred - y_true_v)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true_v) ** 2)))
        acc_sd = float(np.mean(np.abs(y_pred - y_true_v) <= np.maximum(sig_v, 1.0)))
        m = {"spearman": float(rho), "mae": mae, "rmse": rmse, "acc_within_sd": acc_sd, "n": int(len(y_pred)), "valid": len(valid), "total": len(preds)}
        out[tag] = m
        print(f"{tag}: rho={rho:.4f} mae={mae:.4f} rmse={rmse:.4f} acc_sd={acc_sd:.4f} (valid {len(valid)}/{len(preds)})")

    json.dump(out, open(f"{args.out_dir}/eval_metrics.json", "w"), indent=2)
    print(f"Saved metrics to {args.out_dir}/eval_metrics.json")

    if not args.no_shutdown:
        import subprocess
        print("shutting down in 60s ...")
        subprocess.Popen(["sudo", "shutdown", "-h", "+1"])


if __name__ == "__main__":
    main()

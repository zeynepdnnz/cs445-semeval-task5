"""Score AmbiStory with a local instruction-tuned LM using self-consistency.

Uses vLLM if available (batched + fast), falls back to transformers.

Usage:
    python local_lm_score.py \\
        --input data/test_labeled.json \\
        --out results/qwen25_7b_sc5_test.json \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --samples 5 --temperature 0.7
"""

import argparse
import functools
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

print = functools.partial(print, flush=True)

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
    """Robust int-score parser that handles reasoning models with <think>...</think>."""
    if not text:
        return None
    # Strip reasoning blocks (DeepSeek-R1-Distill, Qwen3 in thinking mode).
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)  # unclosed think tag
    text = text.strip()
    if not text:
        return None
    # Prefer the LAST 1-5 digit (final answer in reasoning outputs).
    nums = re.findall(r"\b([1-5])\b", text)
    if nums:
        return int(nums[-1])
    # Fallback: first character if it's a digit.
    if text[0] in "12345":
        return int(text[0])
    return None


def run_vllm(model_id, prompts, samples, temperature, max_tokens=8, max_model_len=2048):
    from vllm import LLM, SamplingParams
    print(f"Loading vLLM model {model_id} ...")
    llm = LLM(model=model_id, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=0.80, enforce_eager=False)
    tokenizer = llm.get_tokenizer()
    chat_prompts = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        try:
            chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        chat_prompts.append(chat)
    sampling = SamplingParams(n=samples, temperature=temperature, max_tokens=max_tokens, top_p=0.95)
    print(f"Generating {len(chat_prompts)} prompts × {samples} samples ...")
    t0 = time.time()
    outputs = llm.generate(chat_prompts, sampling)
    print(f"Generation took {(time.time() - t0) / 60:.1f}m")
    results = []
    for out in outputs:
        scores = [parse_int_score(o.text) for o in out.outputs]
        results.append(scores)
    return results


def run_transformers(model_id, prompts, samples, temperature, max_tokens=8, batch_size=2, encoder_decoder=False):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    print(f"Loading transformers model {model_id} (encoder_decoder={encoder_decoder}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cls = AutoModelForSeq2SeqLM if encoder_decoder else AutoModelForCausalLM
    model = cls.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    if encoder_decoder:
        # FLAN-T5 style: raw prompt, decoder generates the answer fresh.
        chat_prompts = prompts
    else:
        chat_prompts = [
            tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
    results = [[] for _ in chat_prompts]
    for batch_start in range(0, len(chat_prompts), batch_size):
        batch = chat_prompts[batch_start: batch_start + batch_size]
        toks = tokenizer(batch, padding=True, truncation=True, max_length=2048, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = model.generate(
                **toks,
                do_sample=True, temperature=temperature, top_p=0.95,
                max_new_tokens=max_tokens, num_return_sequences=samples,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Decoder-only: outputs include prompt; encoder-decoder: outputs are only the generated tokens.
        if not encoder_decoder:
            outputs = outputs[:, toks["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # group back to batch_size × samples
        for i in range(len(batch)):
            sample_texts = texts[i * samples: (i + 1) * samples]
            results[batch_start + i] = [parse_int_score(t) for t in sample_texts]
        if (batch_start // batch_size) % 10 == 0:
            print(f"  batch {batch_start // batch_size + 1} / {(len(chat_prompts) + batch_size - 1) // batch_size}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--engine", choices=["vllm", "transformers", "auto"], default="auto")
    p.add_argument("--encoder-decoder", action="store_true", help="Use AutoModelForSeq2SeqLM (FLAN-T5 etc.)")
    args = p.parse_args()

    df = pd.read_json(args.input).T
    if args.n is not None:
        df = df.head(args.n)
    prompts = [render_prompt(row) for _, row in df.iterrows()]
    print(f"Loaded {len(prompts)} prompts from {args.input}")

    use_vllm = args.engine == "vllm"
    if args.engine == "auto":
        try:
            import vllm  # noqa
            use_vllm = True
        except ImportError:
            use_vllm = False
    print(f"Engine: {'vLLM' if use_vllm else 'transformers'}")

    if use_vllm and not args.encoder_decoder:
        results_lists = run_vllm(args.model, prompts, args.samples, args.temperature, max_tokens=args.max_tokens)
    else:
        if args.encoder_decoder and use_vllm:
            print("encoder-decoder requested - falling back to transformers")
        results_lists = run_transformers(args.model, prompts, args.samples, args.temperature, max_tokens=args.max_tokens, encoder_decoder=args.encoder_decoder)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    for sid, sample_scores in zip(df.index.astype(str), results_lists):
        valid = [s for s in sample_scores if s is not None]
        cache[str(sid)] = {"mean": float(np.mean(valid)) if valid else None, "samples": sample_scores}
    json.dump(cache, open(args.out, "w"), indent=2)
    print(f"Saved {len(cache)} preds -> {args.out}")

    # Report metrics if labels are present and numeric.
    if df["average"].dtype != object:
        y_pred = np.array([cache[str(k)]["mean"] for k in df.index if cache[str(k)]["mean"] is not None])
        ids_with_pred = [str(k) for k in df.index if cache[str(k)]["mean"] is not None]
        sub = df.loc[[int(i) if str(i).isdigit() else i for i in ids_with_pred]] if df.index.dtype != object else df.loc[ids_with_pred]
        y_true = sub["average"].astype(float).to_numpy()
        sigma = sub["stdev"].astype(float).to_numpy()
        rho, _ = spearmanr(y_pred, y_true)
        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        acc_sd = float(np.mean(np.abs(y_pred - y_true) <= np.maximum(sigma, 1.0)))
        m = {"spearman": float(rho), "mae": mae, "rmse": rmse, "acc_within_sd": acc_sd, "n": int(len(y_pred)), "model": args.model, "samples": args.samples}
        print(f"\n== {args.model} on {Path(args.input).name} ==")
        print(f"  N={m['n']}")
        print(f"  Spearman:      {m['spearman']:.4f}")
        print(f"  MAE:           {m['mae']:.4f}")
        print(f"  RMSE:          {m['rmse']:.4f}")
        print(f"  Acc-within-SD: {m['acc_within_sd']:.4f}")
        json.dump(m, open(args.out.replace(".json", "_metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()

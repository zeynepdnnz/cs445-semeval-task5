"""LoRA fine-tune an instruction-tuned LM on AmbiStory.

Training loss = cross-entropy on the integer rating token. The target is the
human-mean rating rounded to int-1-5 (primary) with optional auxiliary
cross-entropy targeting Gemini's integer score (regularizes toward the strong
LLM teacher).

After training: evaluate on dev + test_labeled with self-consistency
(N samples at temp > 0, mean as the continuous prediction).

Usage:
    python lora_finetune.py \\
        --model Qwen/Qwen3-8B \\
        --out-dir /home/ubuntu/cs445/lora_out/qwen3_8b \\
        --gemini-train /home/ubuntu/cs445/results/gemini_train_soft.json \\
        --epochs 2 --batch-size 4 --grad-accum 4 --lr 2e-4
"""

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import Dataset, DataLoader


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


class AmbiStoryDataset(Dataset):
    """Each item: (prompt_str, target_int).

    If `gemini_labels` is provided and `hybrid=True`, the dataset includes
    BOTH (prompt, human_int) and (prompt, gemini_int) records when they
    differ, effectively giving the LoRA both teacher signals during SFT.
    """

    def __init__(self, df, gemini_labels=None, hybrid=False):
        self.records = []
        for sid, row in df.iterrows():
            prompt = render_prompt(row)
            human_mean = float(row["average"])
            human_int = max(1, min(5, int(round(human_mean))))
            self.records.append({"prompt": prompt, "target_int": human_int, "sigma": float(row["stdev"])})
            if hybrid and gemini_labels is not None:
                g = gemini_labels.get(str(sid))
                if isinstance(g, dict):
                    g = g.get("mean")
                if g is not None:
                    g_int = max(1, min(5, int(round(float(g)))))
                    if g_int != human_int:
                        self.records.append({"prompt": prompt, "target_int": g_int, "sigma": float(row["stdev"])})

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]


def make_collator(tokenizer, max_len=1024):
    def collate(items):
        prompts = []
        targets = []
        for it in items:
            msgs = [{"role": "user", "content": it["prompt"]}]
            try:
                chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(chat)
            targets.append(str(it["target_int"]))
        # Tokenize: prompt + target, masking prompt tokens in labels.
        full_texts = [p + t for p, t in zip(prompts, targets)]
        toks = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        labels = toks["input_ids"].clone()
        # Mask non-target tokens (everything except the last 1-token target).
        prompt_only = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        prompt_lens = (prompt_only["attention_mask"].sum(dim=1)).tolist()
        # Mask everything up to (excluding) the target token positions in labels
        labels[:] = -100
        for i, plen in enumerate(prompt_lens):
            # The target should be at position plen (the next token after the prompt).
            input_ids = toks["input_ids"][i]
            attn = toks["attention_mask"][i]
            seq_len = int(attn.sum().item())
            # Set labels at the first non-prompt position only (the rating token).
            if plen < seq_len:
                labels[i, plen] = input_ids[plen]
        toks["labels"] = labels
        return toks
    return collate


def encode_target_token(tokenizer, digit_str):
    """Return token id corresponding to the single-digit rating."""
    ids = tokenizer.encode(digit_str, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    # Fallback: try " " + digit
    ids2 = tokenizer.encode(" " + digit_str, add_special_tokens=False)
    return ids2[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--train-path", default="/home/ubuntu/cs445/data/train.json")
    p.add_argument("--dev-path", default="/home/ubuntu/cs445/data/dev.json")
    p.add_argument("--test-path", default="/home/ubuntu/cs445/data/test_labeled.json")
    p.add_argument("--gemini-train", default="/home/ubuntu/cs445/results/gemini_train_soft.json")
    p.add_argument("--out-dir", default="/home/ubuntu/cs445/lora_out/qwen3_8b")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--samples", type=int, default=5, help="Self-consistency samples at eval time")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hybrid", action="store_true", help="Augment dataset with Gemini integer targets")
    p.add_argument("--no-shutdown", action="store_true")
    p.add_argument("--task-name", default="lora", help="Tag for logs/output")
    p.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj",
                   help="Comma-separated LoRA target module names. Use 'all_linear' shorthand for attn+MLP.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    tm = args.target_modules
    target_modules = "all-linear" if tm == "all_linear" else [t.strip() for t in tm.split(",") if t.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_df = pd.read_json(args.train_path).T
    dev_df = pd.read_json(args.dev_path).T
    test_df = pd.read_json(args.test_path).T

    gemini_train = None
    if Path(args.gemini_train).exists():
        gemini_train = json.load(open(args.gemini_train))
        print(f"Loaded {len(gemini_train)} Gemini train soft labels")

    train_ds = AmbiStoryDataset(train_df, gemini_labels=gemini_train, hybrid=args.hybrid)
    print(f"Train: {len(train_ds)} examples (hybrid={args.hybrid})")

    collator = make_collator(tokenizer, max_len=args.max_len)
    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    total_steps = math.ceil(len(dl) / args.grad_accum) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)

    model.train()
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(dl):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            if (batch_idx + 1) % args.grad_accum == 0 or batch_idx == len(dl) - 1:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step}/{total_steps} loss={loss.item()*args.grad_accum:.4f} elapsed={time.time()-t0:.0f}s")
    print(f"Training done in {(time.time()-t0)/60:.1f}m")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved LoRA adapter to {args.out_dir}")

    # === Eval ===
    model.config.use_cache = True
    model.eval()

    @torch.inference_mode()
    def score_split(df, tag):
        prompts = [render_prompt(row) for _, row in df.iterrows()]
        chats = [
            tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        token_ids_for_digit = {d: encode_target_token(tokenizer, str(d)) for d in [1, 2, 3, 4, 5]}
        preds = []
        for i in range(0, len(chats), 8):
            batch = chats[i: i + 8]
            toks = tokenizer(batch, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt").to("cuda")
            out = model.generate(
                **toks,
                do_sample=True, temperature=args.temperature, top_p=0.95,
                max_new_tokens=4, num_return_sequences=args.samples,
                pad_token_id=tokenizer.pad_token_id,
            )
            out = out[:, toks["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(out, skip_special_tokens=True)
            for j in range(len(batch)):
                samples = texts[j * args.samples: (j + 1) * args.samples]
                ints = []
                for t in samples:
                    t = t.strip()
                    if t and t[0] in "12345":
                        ints.append(int(t[0]))
                preds.append(float(np.mean(ints)) if ints else None)
            if (i // 8) % 10 == 0:
                print(f"  [{tag}] {i + len(batch)}/{len(chats)}")
        return preds

    out = {}
    for tag, df in [("dev", dev_df), ("test", test_df)]:
        # Skip only if average is a literal placeholder (e.g. '(???)') not a stored-as-object numeric.
        try:
            df["average"].astype(float)
        except (ValueError, TypeError):
            print(f"Skip {tag}: average column not numeric")
            continue
        print(f"Evaluating on {tag} ...")
        preds = score_split(df, tag)
        cache = {}
        for sid, pred in zip(df.index.astype(str), preds):
            cache[sid] = {"mean": pred}
        json.dump(cache, open(f"{args.out_dir}/preds_{tag}.json", "w"), indent=2)
        valid_idx = [i for i, p in enumerate(preds) if p is not None]
        y_pred = np.array([preds[i] for i in valid_idx])
        y_true = df["average"].astype(float).to_numpy()[valid_idx]
        sigma = df["stdev"].astype(float).to_numpy()[valid_idx]
        rho, _ = spearmanr(y_pred, y_true)
        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        acc_sd = float(np.mean(np.abs(y_pred - y_true) <= np.maximum(sigma, 1.0)))
        m = {"spearman": float(rho), "mae": mae, "rmse": rmse, "acc_within_sd": acc_sd, "n": int(len(y_pred))}
        out[tag] = m
        print(f"  {tag}: spearman={rho:.4f} mae={mae:.4f} rmse={rmse:.4f} acc_sd={acc_sd:.4f}")

    json.dump(out, open(f"{args.out_dir}/metrics.json", "w"), indent=2)
    print(f"Saved metrics to {args.out_dir}/metrics.json")

    if not args.no_shutdown:
        import subprocess
        print(f"[{args.task_name}] shutting down in 60s ...")
        subprocess.Popen(["sudo", "shutdown", "-h", "+1"])


if __name__ == "__main__":
    main()

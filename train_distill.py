"""DeBERTa fine-tune on AmbiStory with distillation, ranking, Format B, LLRD.

Drop-in upgrade over the existing notebooks. Improvements vs. baseline:

  1. Hybrid loss = alpha*MSE(human_mean) + beta*MSE(gemini_soft) + gamma*ranking
     Setting beta=0 reduces to a no-Gemini model. Setting gamma=0 disables ranking.
  2. Within-context pairwise ranking on samples that share (precontext, sentence, ending).
     Directly optimizes Spearman, which MSE alone does not.
  3. Format B: optional `<<homonym>>` markers in the ambiguous sentence (GlossBERT-style).
  4. Layer-wise learning rate decay (LLRD): standard recipe for large-encoder
     fine-tuning, stabilizes top-layer adaptation.
  5. Annotator-disagreement weighting: examples with high std contribute less to MSE.
  6. Multi-seed runs; reports test mean +/- std.

Colab usage:
    !pip install -q "transformers>=4.40,<4.45"
    from train_distill import run

    run(
        train_path='/content/train.json',
        dev_path='/content/dev.json',
        test_path='/content/test.json',
        gemini_train_path='/content/drive/MyDrive/gemini_train_soft.json',  # optional
        out_dir='/content/distill_out',
        seeds=[42, 1337, 2024],
        mark_homonym=True,     # Format B
        alpha=0.5, beta=0.5, gamma=0.3,
        sigma_weighting=True,
        llrd_decay=0.9,
    )
"""

import gc
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, Value
from scipy.stats import spearmanr
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

DEFAULT_MODEL_ID = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
MODEL_ID = DEFAULT_MODEL_ID  # may be overridden via run(base_model=...)


# ---------------------------------------------------------------------------
# Data loading + tokenization
# ---------------------------------------------------------------------------

def _mark(sentence: str, word: str) -> str:
    """Wrap the first occurrence of `word` in `sentence` with << ... >> markers.

    Falls back to leaving the sentence unchanged if the word is not found
    (homonym surface form may have been morphologically inflected — rare in
    AmbiStory but handled defensively).
    """
    pattern = re.compile(r"\b" + re.escape(word) + r"\w*\b", re.IGNORECASE)
    m = pattern.search(sentence)
    if not m:
        return sentence
    s, e = m.span()
    return sentence[:s] + "<<" + sentence[s:e] + ">>" + sentence[e:]


def _row_to_pair(row: dict, mark_homonym: bool) -> Tuple[str, str]:
    ending = row["ending"] if isinstance(row["ending"], str) and row["ending"] else ""
    sentence = row["sentence"]
    if mark_homonym:
        sentence = _mark(sentence, row["homonym"])
    story = f"{row['precontext']} {sentence} {ending}".strip()
    meaning = (
        f"{row['homonym']}: {row['judged_meaning']} "
        f'(e.g., "{row["example_sentence"]}")'
    )
    return story, meaning


def _context_key(row: dict) -> str:
    """Identifier for the shared story across multiple senses. Used to form
    within-context ranking groups."""
    return "|".join([
        (row["precontext"] or "").strip(),
        (row["sentence"] or "").strip(),
        (row["ending"] or "").strip() if row["ending"] else "",
    ])


def load_splits(
    train_path: str,
    dev_path: str,
    test_path: str,
    gemini_train_path: Optional[str] = None,
    mark_homonym: bool = False,
) -> DatasetDict:
    splits = {}
    for name, path in [("train", train_path), ("validation", dev_path), ("test", test_path)]:
        df = pd.read_json(path).T
        df["context_id"] = df.apply(_context_key, axis=1)
        df["labels"] = df["average"].astype("float32")
        df["sigma"] = df["stdev"].astype("float32") if "stdev" in df.columns else np.float32(0.5)
        if name == "train" and gemini_train_path:
            with open(gemini_train_path) as f:
                gem_raw = {str(k): v for k, v in json.load(f).items()}
            # Handle two formats: float (old) or {"mean": float, "samples": [...]} (new SC).
            gem = {}
            for k, v in gem_raw.items():
                if isinstance(v, dict):
                    gem[k] = v.get("mean")
                else:
                    gem[k] = v
            df["gemini_label"] = df.index.astype(str).map(lambda k: gem.get(k))
            n_missing = df["gemini_label"].isna().sum()
            if n_missing:
                print(f"  warning: {n_missing}/{len(df)} train rows missing Gemini labels — fallback to human mean")
                df["gemini_label"] = df["gemini_label"].fillna(df["labels"])
            df["gemini_label"] = df["gemini_label"].astype("float32")
        else:
            df["gemini_label"] = df["labels"]
        splits[name] = df
    return splits


def make_tokenized(splits, tokenizer, mark_homonym: bool, max_length: int = 256):
    def to_pair(row):
        return _row_to_pair(row, mark_homonym)

    ds_dict = {}
    for name, df in splits.items():
        records = []
        # Map each context_id to a contiguous integer group_id so we can do
        # within-context ranking inside the model's compute_loss.
        ctx_to_gid = {c: i for i, c in enumerate(df["context_id"].unique())}
        for _, row in df.iterrows():
            story, meaning = to_pair(row.to_dict())
            enc = tokenizer(story, meaning, truncation="only_first", max_length=max_length, padding=False)
            records.append({
                **{k: v for k, v in enc.items()},
                "labels": float(row["labels"]),
                "gemini_label": float(row["gemini_label"]),
                "sigma": float(row["sigma"]),
                "group_id": int(ctx_to_gid[row["context_id"]]),
            })
        ds_dict[name] = Dataset.from_list(records)
    return DatasetDict(ds_dict)


# ---------------------------------------------------------------------------
# Custom collator that preserves per-example float fields + group_id
# ---------------------------------------------------------------------------

class DistillCollator:
    def __init__(self, tokenizer):
        self.base = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        extras = {k: torch.tensor([f[k] for f in features]) for k in ("gemini_label", "sigma", "group_id")}
        stripped = [{k: v for k, v in f.items() if k not in extras} for f in features]
        batch = self.base(stripped)
        batch.update(extras)
        return batch


# ---------------------------------------------------------------------------
# Distillation Trainer
# ---------------------------------------------------------------------------

@dataclass
class LossConfig:
    alpha: float = 1.0          # human-mean MSE
    beta: float = 0.0           # Gemini soft-label MSE
    gamma: float = 0.0          # within-context pairwise ranking
    margin: float = 0.3         # ranking margin (in label units, 1-5 scale)
    sigma_weighting: bool = False
    sigma_floor: float = 0.3    # avoid 1/0 weights on perfect-agreement examples


class DistillTrainer(Trainer):
    def __init__(self, *args, loss_config: LossConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_config = loss_config

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        cfg = self.loss_config
        labels = inputs.pop("labels")
        gemini = inputs.pop("gemini_label")
        sigma = inputs.pop("sigma")
        groups = inputs.pop("group_id")

        outputs = model(**inputs)
        preds = outputs.logits.squeeze(-1)

        # Per-example weights for the human MSE term.
        if cfg.sigma_weighting:
            w = 1.0 / torch.clamp(sigma, min=cfg.sigma_floor)
            w = w / w.mean()
        else:
            w = torch.ones_like(preds)

        loss = 0.0
        if cfg.alpha:
            loss = loss + cfg.alpha * (w * (preds - labels) ** 2).mean()
        if cfg.beta:
            loss = loss + cfg.beta * ((preds - gemini) ** 2).mean()
        if cfg.gamma:
            loss = loss + cfg.gamma * self._rank_loss(preds, labels, groups, cfg.margin)

        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _rank_loss(preds, labels, groups, margin):
        """Margin-based pairwise rank loss inside each group_id.

        For each pair (i, j) sharing a context where labels[i] > labels[j]:
            loss += max(0, margin*(labels[i]-labels[j]) - (preds[i]-preds[j]))
        """
        device = preds.device
        loss_terms = []
        unique_groups = torch.unique(groups)
        for g in unique_groups:
            mask = groups == g
            if mask.sum() < 2:
                continue
            p = preds[mask]
            y = labels[mask]
            n = p.shape[0]
            diff_p = p.unsqueeze(0) - p.unsqueeze(1)         # (n,n) pred margins
            diff_y = y.unsqueeze(0) - y.unsqueeze(1)         # (n,n) label margins
            # Only count pairs where label[i] > label[j].
            valid = (diff_y > 0).float()
            target_margin = margin * diff_y                  # scaled margin per pair
            hinge = torch.clamp(target_margin - diff_p, min=0.0) * valid
            denom = valid.sum().clamp(min=1.0)
            loss_terms.append(hinge.sum() / denom)
        if not loss_terms:
            return torch.tensor(0.0, device=device)
        return torch.stack(loss_terms).mean()


# ---------------------------------------------------------------------------
# LLRD optimizer
# ---------------------------------------------------------------------------

def build_llrd_optimizer(model, base_lr: float, head_lr: float, decay: float, weight_decay: float):
    """Decreasing LR from top encoder layer down to embeddings; head gets head_lr."""
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = []

    # Head (classifier + pooler).
    head_named = [(n, p) for n, p in model.named_parameters() if "classifier" in n or "pooler" in n]
    for n, p in head_named:
        params.append({
            "params": [p],
            "lr": head_lr,
            "weight_decay": 0.0 if any(nd in n for nd in no_decay) else weight_decay,
        })

    # Encoder layers.
    layer_pat = re.compile(r"encoder\.layer\.(\d+)\.")
    layer_groups: Dict[int, List[Tuple[str, torch.nn.Parameter]]] = {}
    embedding_params = []
    for n, p in model.named_parameters():
        if any(n == hn for hn, _ in head_named):
            continue
        m = layer_pat.search(n)
        if m:
            layer_groups.setdefault(int(m.group(1)), []).append((n, p))
        elif "embeddings" in n:
            embedding_params.append((n, p))

    if layer_groups:
        max_layer = max(layer_groups)
        for idx, ps in layer_groups.items():
            lr = base_lr * (decay ** (max_layer - idx))
            for n, p in ps:
                params.append({
                    "params": [p],
                    "lr": lr,
                    "weight_decay": 0.0 if any(nd in n for nd in no_decay) else weight_decay,
                })
        emb_lr = base_lr * (decay ** (max_layer + 1))
    else:
        emb_lr = base_lr

    for n, p in embedding_params:
        params.append({
            "params": [p],
            "lr": emb_lr,
            "weight_decay": 0.0 if any(nd in n for nd in no_decay) else weight_decay,
        })
    return torch.optim.AdamW(params)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metric_fn(eval_pred: EvalPrediction):
    preds, labels = eval_pred
    preds = preds.squeeze()
    rho, _ = spearmanr(preds, labels)
    mae = float(np.mean(np.abs(preds - labels)))
    rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))
    return {"spearman": float(rho), "mae": mae, "rmse": rmse}


def acc_within_sd(preds, labels, sigmas, floor=1.0):
    threshold = np.maximum(sigmas, floor)
    return float(np.mean(np.abs(preds - labels) <= threshold))


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def _train_one(
    tokenized,
    tokenizer,
    test_sigmas,
    seed: int,
    out_dir: str,
    loss_config: LossConfig,
    base_lr: float,
    head_lr: float,
    llrd_decay: Optional[float],
    epochs: int,
    batch_size: int,
    warmup_ratio: float,
    weight_decay: float,
    early_stopping: int,
):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=1, problem_type="regression", ignore_mismatched_sizes=True,
    )
    # Enable gradient checkpointing for larger encoders to fit A10G.
    use_grad_ckpt = "xxlarge" in MODEL_ID.lower() or "xx-large" in MODEL_ID.lower()
    args = TrainingArguments(
        output_dir=f"{out_dir}/seed{seed}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=base_lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        bf16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="spearman",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
        seed=seed,
        remove_unused_columns=False,
        gradient_checkpointing=use_grad_ckpt,
    )
    if use_grad_ckpt:
        model.gradient_checkpointing_enable()
    optimizers = (None, None)
    if llrd_decay is not None:
        optimizer = build_llrd_optimizer(model, base_lr=base_lr, head_lr=head_lr, decay=llrd_decay, weight_decay=weight_decay)
        optimizers = (optimizer, None)

    # transformers >=4.46 renamed tokenizer= to processing_class=.
    import transformers as _tr
    from packaging.version import Version as _V
    _proc_key = "processing_class" if _V(_tr.__version__) >= _V("4.46") else "tokenizer"
    trainer_kwargs = {
        "model": model,
        "args": args,
        "data_collator": DistillCollator(tokenizer),
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "compute_metrics": metric_fn,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=early_stopping)] if early_stopping else [],
        "loss_config": loss_config,
        "optimizers": optimizers,
        _proc_key: tokenizer,
    }
    trainer = DistillTrainer(**trainer_kwargs)
    trainer.train()

    dev = trainer.evaluate(tokenized["validation"])
    test = trainer.evaluate(tokenized["test"])
    test_preds = trainer.predict(tokenized["test"]).predictions.squeeze()
    test_labels = np.array(tokenized["test"]["labels"], dtype=float)
    test["acc_within_sd"] = acc_within_sd(test_preds, test_labels, test_sigmas)

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"dev": dev, "test": test, "test_preds": test_preds.tolist()}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    train_path: str,
    dev_path: str,
    test_path: str,
    out_dir: str,
    gemini_train_path: Optional[str] = None,
    seeds: List[int] = (42, 1337, 2024),
    mark_homonym: bool = True,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.3,
    margin: float = 0.3,
    sigma_weighting: bool = True,
    base_lr: float = 1e-5,
    head_lr: float = 1e-4,
    llrd_decay: Optional[float] = 0.9,
    epochs: int = 5,
    batch_size: int = 8,
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.01,
    early_stopping: int = 2,
    max_length: int = 256,
    base_model: Optional[str] = None,
):
    global MODEL_ID
    if base_model is not None:
        MODEL_ID = base_model
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if beta and not gemini_train_path:
        print("beta>0 but no gemini_train_path given — forcing beta=0")
        beta = 0.0

    splits = load_splits(train_path, dev_path, test_path, gemini_train_path, mark_homonym)
    test_sigmas = splits["test"]["sigma"].to_numpy(dtype=float)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenized = make_tokenized(splits, tokenizer, mark_homonym=mark_homonym, max_length=max_length)

    loss_config = LossConfig(
        alpha=alpha, beta=beta, gamma=gamma, margin=margin, sigma_weighting=sigma_weighting,
    )
    print(f"Loss: alpha={alpha} beta={beta} gamma={gamma} margin={margin} sigma_weighting={sigma_weighting}")
    print(f"LLRD decay={llrd_decay}  base_lr={base_lr}  head_lr={head_lr}")
    print(f"Format B (homonym markers): {mark_homonym}")
    print(f"Seeds: {list(seeds)}\n")

    results = []
    for seed in seeds:
        print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")
        res = _train_one(
            tokenized, tokenizer, test_sigmas,
            seed=seed, out_dir=out_dir, loss_config=loss_config,
            base_lr=base_lr, head_lr=head_lr, llrd_decay=llrd_decay,
            epochs=epochs, batch_size=batch_size, warmup_ratio=warmup_ratio,
            weight_decay=weight_decay, early_stopping=early_stopping,
        )
        results.append({"seed": seed, **res})
        print(
            f"  seed={seed}  "
            f"dev rho={res['dev']['eval_spearman']:.4f}  "
            f"test rho={res['test']['eval_spearman']:.4f}  "
            f"test acc_sd={res['test']['acc_within_sd']:.4f}"
        )

    test_rhos = np.array([r["test"]["eval_spearman"] for r in results])
    test_acc = np.array([r["test"]["acc_within_sd"] for r in results])
    summary = {
        "test_spearman_mean": float(test_rhos.mean()),
        "test_spearman_std": float(test_rhos.std(ddof=0)),
        "test_acc_within_sd_mean": float(test_acc.mean()),
        "test_acc_within_sd_std": float(test_acc.std(ddof=0)),
        "per_seed": [
            {
                "seed": r["seed"],
                "dev_spearman": float(r["dev"]["eval_spearman"]),
                "test_spearman": float(r["test"]["eval_spearman"]),
                "test_mae": float(r["test"]["eval_mae"]),
                "test_rmse": float(r["test"]["eval_rmse"]),
                "test_acc_within_sd": float(r["test"]["acc_within_sd"]),
            }
            for r in results
        ],
        "config": {
            "alpha": alpha, "beta": beta, "gamma": gamma, "margin": margin,
            "sigma_weighting": sigma_weighting, "mark_homonym": mark_homonym,
            "base_lr": base_lr, "head_lr": head_lr, "llrd_decay": llrd_decay,
            "epochs": epochs, "batch_size": batch_size,
            "warmup_ratio": warmup_ratio, "weight_decay": weight_decay,
        },
    }
    print(f"\n{'='*60}\nTEST SET (mean across {len(seeds)} seeds)\n{'='*60}")
    print(f"  Spearman:      {summary['test_spearman_mean']:.4f} +/- {summary['test_spearman_std']:.4f}")
    print(f"  Acc-within-SD: {summary['test_acc_within_sd_mean']:.4f} +/- {summary['test_acc_within_sd_std']:.4f}")

    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {out_dir}/summary.json")
    return summary

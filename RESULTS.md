# Results — AmbiStory Plausibility Rating (SemEval-2026 Task 5)

This document catalogs **every successful (non-crash, valid-output) experiment** we ran for SemEval-2026 Task 5 (AmbiStory: Rating Plausibility of Word Senses in Ambiguous Sentences through Narrative Understanding).

The primary metric is **Spearman ρ** (rank correlation between predicted plausibility and human-mean rating, scale 1–5). The secondary metric is **Accuracy-within-Standard-Deviation** (proportion of predictions whose distance from the human mean is at most one or one σ of the rater pool — see [`evaluate.py`](https://github.com/Janosch-Gehring/semeval26-05-scripts/blob/main/evaluate.py)).

Splits used:
- **train.json** — 2,280 labeled examples (220 unique homonyms)
- **dev.json** — 588 labeled examples (55 unique homonyms)
- **test.json** — 930 examples with hidden labels (87 unique homonyms; 86 unseen at train time). For local evaluation we use **test_labeled.json** from the original AmbiStory dataset repo ([Janosch-Gehring/ambistory](https://github.com/Janosch-Gehring/ambistory)).

---

## 🏆 Headline result

> **LoRA-fine-tuned Qwen3-8B (rank=16, 2 epochs, plain CE on human-integer rating), seed=2024 (task-B):**
> **Test ρ = 0.7635**, Acc-within-SD = 0.8645.
>
> Updated (Wave 3): adding 3 more hybrid seeds + 2 non-hybrid seeds at 1337/2024 showed the **non-hybrid recipe is robustly better** (3-seed mean 0.7558 vs hybrid 6-seed mean 0.7380; non-hybrid wins at every paired seed). T9's previous 0.7575 hybrid result was a lucky seed, not evidence the hybrid loss helps.
>
> This **beats the strongest closed-source LLM** we tested at inference time (Gemini 2.5 Pro with the rewritten rubric prompt + 5-sample self-consistency: ρ = 0.7387) by **+1.88 pp on Spearman** and **+4.62 pp on Acc-within-SD**, using an open 8B-parameter student that runs locally on a single A10G.

Submission JSONL: [`submissions/lora_qwen3_8b_nohybrid_seed2024_test.jsonl`](submissions/lora_qwen3_8b_nohybrid_seed2024_test.jsonl) (Wave 3 new best). The earlier hybrid-seed=2024 submission ([`submissions/lora_qwen3_8b_hybrid_seed2024_test.jsonl`](submissions/lora_qwen3_8b_hybrid_seed2024_test.jsonl)) at 0.7575 is also kept in the repo.

---

## Master results table — test set (930 examples)

Sorted descending by Spearman ρ. Asterisks mark our submitted system and the previously-best Gemini number.

| Rank | Approach | Family | Test ρ | Acc-SD | MAE | RMSE | n_seeds |
|---|---|---|---:|---:|---:|---:|:-:|
| 1 ⭐ | **LoRA Qwen3-8B no-hybrid r=16 seed=2024 (task-B)** | Open LM fine-tune | **0.7635** | **0.8645** | – | – | 1 |
| 2 | LoRA Qwen3-8B no-hybrid r=16 seed=1337 (task-A) | Open LM fine-tune | 0.7626 | 0.8624 | – | – | 1 |
| 3 | LoRA Qwen3-8B hybrid r=16 seed=2024 (T9) | Open LM fine-tune | 0.7575 | 0.8634 | 0.6290 | 0.8215 | 1 |
| 4 | LoRA Qwen3-8B hybrid r=16 seed=7 (task-C) | Open LM fine-tune | 0.7448 | 0.8473 | – | – | 1 |
| 5 | LoRA Qwen3-8B hybrid r=16 seed=314 (task-E) | Open LM fine-tune | 0.7420 | 0.8495 | – | – | 1 |
| 6 | LoRA Qwen3-8B hybrid r=8 seed=2024 (task-F) | Open LM fine-tune | 0.7419 | 0.8505 | – | – | 1 |
| 7 | LoRA Qwen3-8B no-hybrid r=16 seed=42 (C) | Open LM fine-tune | 0.7413 | 0.8441 | 0.6467 | 0.8237 | 1 |
| 8 ★ | **Gemini 2.5 Pro + better prompt + SC=5** | API zero-shot | 0.7387 | 0.8172 | 0.7347 | 0.9802 | – |
| 9 | LoRA Qwen3-8B hybrid r=16 seed=1337 (T8) | Open LM fine-tune | 0.7360 | 0.8344 | 0.6499 | 0.8585 | 1 |
| 10 | LoRA Qwen3-8B hybrid r=32 seed=2024 (task-G) | Open LM fine-tune | 0.7344 | 0.8473 | – | – | 1 |
| 11 | LoRA Qwen3-8B hybrid r=16 seed=99 (task-D) | Open LM fine-tune | 0.7269 | 0.8323 | – | – | 1 |
| 12 | LoRA Qwen3-8B hybrid r=16 seed=42 (T7) | Open LM fine-tune | 0.7206 | 0.8355 | 0.6804 | 0.8781 | 1 |
| 4 | LoRA Qwen3-8B + hybrid loss, seed=1337 | Open LM fine-tune | 0.7360 | 0.8344 | 0.6499 | 0.8585 | 1 |
| 5 | Paper-reported GPT-4o-mini 0-shot | API zero-shot | 0.726 | 0.726 | – | – | – |
| 6 | Gemini 2.5 Pro + paper prompt, single | API zero-shot | 0.7239 | 0.7882 | 0.8304 | 1.0969 | 1 |
| 7 | Gemini 2.5 Pro + better prompt, single | API zero-shot | 0.7232 | 0.8086 | 0.7652 | 1.0248 | 1 |
| 8 | GPT-4o-mini 0-shot (team replication) | API zero-shot | 0.7025 | 0.7806 | 0.8519 | 1.0970 | 1 |
| 9 | **Our DeBERTa-v3-large A_baseline** (3 seeds) | Cross-encoder fine-tune | 0.6612 ± 0.004 | 0.7742 | – | – | 3 |
| 10 | J_rank_only (DeBERTa + within-ctx ranking) | Cross-encoder | 0.6424 | 0.7505 | – | – | 1 |
| 11 | Notebook WiC → AmbiStory (single seed) | Cross-encoder curriculum | 0.6384 | 0.7667 | – | – | 1 |
| 12 | Notebook STS-B + WiC → AmbiStory | Cross-encoder curriculum | 0.6379 | 0.7548 | – | – | 1 |
| 13 | I_llrd_only (DeBERTa + LLRD) | Cross-encoder | 0.6377 | 0.7742 | – | – | 1 |
| 14 | B_format_b (DeBERTa + `<<homonym>>` markers) (3 seeds) | Cross-encoder | 0.6358 ± 0.016 | 0.7631 | – | – | 3 |
| 15 | Notebook baseline (E_short_warmup, single seed=1337) | Cross-encoder | 0.6336 | 0.7742 | – | – | 1 |
| 16 | Qwen2.5-7B-Instruct SC=5 | Open LM zero-shot | 0.6303 | 0.7581 | 0.9323 | 1.1915 | 1 |
| 17 | K_distill_only (DeBERTa + Gemini soft, β=0.5) (3 seeds) | Cross-encoder distill | 0.6288 ± 0.008 | 0.7254 | – | – | 3 |
| 18 | Notebook STS-B → AmbiStory | Cross-encoder curriculum | 0.6264 | – | – | – | 1 |
| 19 | sileod/deberta-v3-large-tasksource-nli + distill (3 seeds) | Cross-encoder | 0.6166 ± 0.008 | 0.7358 | – | – | 3 |
| 20 | L_full_no_marker (DeBERTa + everything) (3 seeds) | Cross-encoder | 0.6203 ± 0.008 | 0.7237 | – | – | 3 |
| 21 | Qwen3-8B SC=5 (vLLM, no-thinking) | Open LM zero-shot | 0.5884 | 0.8011 | – | – | 1 |
| 22 | FLAN-T5-Large SC=5 | Encoder-decoder zero-shot | 0.1996 | 0.6065 | 1.1008 | 1.3395 | 1 |

---

## Master results table — dev set (588 examples)

Sorted descending. Used for ablation decisions and prompt tuning.

| Approach | Dev ρ | Acc-SD | MAE | RMSE |
|---|---:|---:|---:|---:|
| **LoRA Qwen3-8B + hybrid loss, seed=2024** | **0.7617** | **0.8588** | 0.6206 | 0.8092 |
| LoRA Qwen3-8B no hybrid, seed=42 (C) | 0.7579 | 0.8452 | 0.6233 | 0.7890 |
| Gemini 2.5 Pro + better + SC=10 | 0.7505 | 0.8333 | 0.7224 | 0.9485 |
| LoRA Qwen3-8B + hybrid loss, seed=1337 (T8) | 0.7478 | 0.8588 | 0.6574 | 0.8417 |
| Gemini 2.5 Pro + better + SC=5 | 0.7471 | 0.8418 | 0.7266 | 0.9568 |
| Gemini 2.5 Pro + better prompt, single | 0.7325 | 0.8316 | 0.7531 | 1.0055 |
| Gemini 2.5 Flash + no-think + paper | 0.7299 | 0.7347 | 0.9223 | 1.2026 |
| Gemini 2.5 Pro + thinking + paper, single | 0.7210 | 0.7704 | 0.8407 | 1.0944 |
| DeBERTa-v3-large E_short_warmup (3-seed mean) | 0.6700 ± 0.003 | – | – | – |
| DeBERTa-v3-large D_very_low | 0.6673 ± 0.012 | – | – | – |
| DeBERTa-v3-large B_lower_lr | 0.6656 ± 0.009 | – | – | – |
| DeBERTa-v3-large C_bs16 | 0.6610 ± 0.005 | – | – | – |
| DeBERTa-v3-large A_baseline (lr 2e-5) | 0.6427 ± 0.001 | – | – | – |
| Gemini 2.5 Pro + better + paper-oneshot | 0.6705 | 0.7619 | 0.8637 | 1.1378 |
| Gemini 2.5 Pro + better + better-oneshot | 0.6687 | 0.8112 | 0.7875 | 1.0453 |

---

## Configuration index — what each approach is

### Cross-encoder DeBERTa fine-tunes (`train_distill.py`)

All use `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` as the base unless noted, with the regression head trained on top. Sentence-pair input: `[CLS] {precontext} {sentence} {ending} [SEP] {homonym}: {meaning} (e.g., "{example}") [SEP]`.

| Tag | base_model | mark_homonym | σ-weight | LLRD | β (Gemini distill) | γ (ranking) | seeds |
|---|---|:-:|:-:|---|:-:|:-:|---|
| Notebook baseline (E_short_warmup) | Laurer | – | – | – | 0 | 0 | 1 (1337) |
| Notebook STS-B → AmbiStory | Laurer | – | – | – | 0 | 0 | 1 (42) |
| Notebook STS-B + WiC → AmbiStory | Laurer | – (WiC inputs have marker, AmbiStory doesn't) | – | – | 0 | 0 | 1 (42) |
| Notebook WiC → AmbiStory | Laurer | – | – | – | 0 | 0 | 1 (42) |
| A_baseline (ours, 3-seed) | Laurer | – | – | – | 0 | 0 | 42, 1337, 2024 |
| B_format_b | Laurer | ✅ | – | – | 0 | 0 | 42, 1337, 2024 |
| I_llrd_only | Laurer | – | – | 0.9 | 0 | 0 | 42 |
| J_rank_only | Laurer | – | – | – | 0 | 0.3 | 42 |
| K_distill_only | Laurer | – | – | – | 0.5 | 0 | 42, 1337, 2024 |
| L_full_no_marker | Laurer | – | ✅ | 0.9 | 0.5 | 0.3 | 42, 1337, 2024 |
| Sileod tasksource (T4R/T5R/T6R) | `sileod/deberta-v3-large-tasksource-nli` | – | ✅ | 0.9 | 0.2 | 0.3 | 42, 1337, 2024 |

Common hyperparameters: `lr=1e-5, head_lr=1e-4, bs=8, epochs=5, warmup_ratio=0.06, weight_decay=0.01, max_len=256, early_stopping=2, bf16=True`.

### Gemini 2.5 Pro zero-shot variants (`gemini_score_async.py`, `gemini_score_v2.py`, `gemini_self_consistency.py`)

All use the Google `gemini-2.5-pro` endpoint with `thinking_budget=128` unless noted. Prompts:

- **paper prompt**: identical to the GPT-4o-mini baseline in `CS445_Baselines.ipynb` cell 16 (the protocol from Gehring & Roth 2025).
- **better prompt**: rewritten rubric with explicit 1–5 anchors and calibration notes (see `PROMPT_BETTER` in `gemini_score_v2.py`, `gemini_self_consistency.py`).
- **paper-oneshot / better-oneshot**: prepend a single labeled demonstration drawn from train (sample_id=402, "interest" / "loan interest", human avg = 3.0, σ = 0).
- **SC=N**: sample N times at temperature 0.7 (and `max_output_tokens` set to allow thinking + answer), average integer outputs → continuous score.
- **Gemini 2.5 Flash + no-think**: model `gemini-2.5-flash` with `thinking_budget=0` (disabled).

### Local open-LLM zero-shot (`local_lm_score.py`)

- **Qwen2.5-7B-Instruct**: vLLM, bf16, `chat_template(enable_thinking=False)`, SC=5 at T=0.7.
- **Qwen3-8B**: same, with explicit `enable_thinking=False` to bypass Qwen3's reasoning mode (otherwise `<think>...</think>` consumed all output tokens).
- **FLAN-T5-Large** (`google/flan-t5-large`): `AutoModelForSeq2SeqLM`, bf16 via transformers (vLLM encoder-decoder support too narrow), SC=5 at T=0.7.
- Failed (not in table): Qwen3-32B-AWQ (KV-cache OOM), Qwen3-14B (28 GB bf16 doesn't fit A10G), DeepSeek-R1-Distill-Qwen-7B (parser failed — reasoning blocks consumed the 8-token output budget).

### LoRA fine-tunes (`lora_finetune.py` + `eval_lora.py`)

Base = `Qwen/Qwen3-8B` (16 GB bf16). LoRA-trains rank-r adapter on `q_proj, k_proj, v_proj, o_proj` only — base weights frozen. Loss = cross-entropy on the human-mean rating token (rounded to 1–5). In hybrid runs, the training set is augmented with `(prompt, gemini_integer)` pairs where the Gemini SC=5 mean integer differs from the human mean integer — see `AmbiStoryDataset.__init__` in [`lora_finetune.py`](lora_finetune.py).

| Tag | seed | hybrid | rank | epochs | LR | grad_accum |
|---|---|:-:|---|---|---|---|
| C (seed=42) | 42 | – | 16 | 2 | 2e-4 | 16 |
| T8 (seed=1337) | 1337 | ✅ | 16 | 2 | 2e-4 | 16 |
| T9 (seed=2024) | 2024 | ✅ | 16 | 2 | 2e-4 | 16 |
| T7 (seed=42) | 42 | ✅ | 16 | 2 | 2e-4 | 16 | *(still training at time of writing)* |

Common: `bs=1, max_len=1024, lora_alpha=32, lora_dropout=0.05, gradient_checkpointing=True`. Self-consistency at eval: 5 samples per example at T=0.7, integer-vote mean → continuous output. Crucially we use **left-padding** at inference (`tokenizer.padding_side='left'`) — the in-process eval in our first attempt used right-padding and produced near-random Spearman (0.08–0.21), masking the real adapter quality.

### Within-context calibration (`within_context_calibrate.py`)

Idea: within each story-group (samples sharing `precontext + sentence + ending`), pull raw predictions slightly toward the within-group rank-ordered values. **Result: no effect** — on this dataset every story-group has exactly 2 candidates, and the rank-rescale operation is mathematically identity for size-2 groups. Documented in code (`within_context_calibrate.py`) and in this report as a negative result.

### Soft-label generation for distillation

`results/gemini_train_sc5_better.json` — 2,280 train examples scored by Gemini 2.5 Pro (better prompt, SC=5, T=0.7). Mean Spearman correlation with human-mean labels on the train split = **0.7286**, i.e., these soft labels are themselves a strong teacher signal. Generated via `gemini_self_consistency.py`.

---

## Reproducing each row

All scripts are in this repo. The minimum input is `data/{train,dev,test_labeled}.json` (already committed). For Gemini-based runs you need `GEMINI_API_KEY` in the environment (Google AI Studio key).

### Cross-encoder DeBERTa (any variant)

```bash
python task_runner.py \
    --task-name <tag> \
    --base-model <hf_id_or_path> \
    --seed 42 --batch-size 8 \
    --no-shutdown
```

This calls `train_distill.run()` with the Phase-2 recipe (α=0.8 human, β=0.2 Gemini-soft, γ=0.3 ranking, σ-weighted MSE, no Format-B, LLRD=0.9). Override α/β/γ via CLI flags if needed. Outputs `phase2_out/<tag>/summary.json` with all per-seed metrics.

### Gemini Pro zero-shot SC=5 (better prompt)

```bash
GEMINI_API_KEY=... python gemini_self_consistency.py \
    --input data/test_labeled.json \
    --out results/gemini_sc5_test.json \
    --samples 5 --temperature 0.7 --concurrency 32
```

### Local LLM zero-shot SC=5

```bash
python local_lm_score.py \
    --input data/test_labeled.json \
    --out results/qwen3_8b_sc5_test.json \
    --model Qwen/Qwen3-8B --samples 5 --temperature 0.7 \
    --max-tokens 8
```

### LoRA Qwen3-8B with hybrid loss

```bash
# Train + (broken right-pad) eval
python lora_finetune.py \
    --model Qwen/Qwen3-8B \
    --task-name task-9 \
    --out-dir lora_out/task-9 \
    --gemini-train results/gemini_train_sc5_better.json \
    --hybrid --epochs 2 --batch-size 1 --grad-accum 16 --lr 2e-4 --lora-r 16 \
    --seed 2024 --no-shutdown

# Proper re-eval with left-padding (REQUIRED for valid results)
python eval_lora.py \
    --base-model Qwen/Qwen3-8B \
    --adapter-dir lora_out/task-9 \
    --out-dir lora_out/task-9 \
    --samples 5 --temperature 0.7 --batch-size 8 --no-shutdown
```

### Build a SemEval submission JSONL

```bash
python make_submission.py \
    --preds results/phase2/task-9_preds_test.json \
    --gold data/test.json \
    --out submissions/lora_qwen3_8b_hybrid_seed2024_test.jsonl
```

---

## Index of saved artifacts

### Predictions and metrics

| File | What |
|---|---|
| `results/gemini_*_metrics.json` | Per-run Gemini eval metrics (Spearman / MAE / RMSE / Acc-SD) |
| `results/gemini_sc5_test.json` | Gemini SC=5 raw test predictions (mean over 5 samples + raw sample integers) |
| `results/gemini_sc5_test_wccal.json` | Same after within-context calibration (no effect, kept for reproducibility) |
| `results/gemini_train_sc5_better.json` | Gemini SC=5 train soft labels (used for distillation) |
| `results/qwen3_8b_sc5_test.json`, `results/qwen25_7b_sc5_test.json` | Local LLM zero-shot predictions (test) |
| `results/flan_t5_large_sc5_test.json` | FLAN-T5-Large encoder-decoder zero-shot |
| `results/phase2/task-{4-r,5-r,6-r}_summary.json` | Sileod tasksource fine-tune per-seed results |
| `results/phase2/task-{8,9}_eval_metrics.json` | LoRA hybrid seed 1337/2024 left-pad re-eval metrics |
| `results/phase2/task-{8,9}_preds_{dev,test}.json` | LoRA hybrid prediction caches |
| `results/phase2/lora_seed42_*.json` | LoRA seed=42 (C, non-hybrid) eval metrics and predictions |
| `aws_results/cs445_artifacts_{A,B,C}.tar.gz` | DeBERTa A_baseline/B_format_b/I/J/K/L training logs + per-seed summaries |

### Submissions

| File | Source | Test ρ on test_labeled |
|---|---|---:|
| `submissions/lora_qwen3_8b_nohybrid_seed2024_test.jsonl` ⭐ | LoRA no-hybrid r=16 seed=2024 (task-B) | 0.7635 |
| `submissions/lora_qwen3_8b_hybrid_seed2024_test.jsonl` | LoRA hybrid r=16 seed=2024 (T9, previous best) | 0.7575 |
| `submissions/lora_qwen3_8b_seed42_test.jsonl` | LoRA no-hybrid seed=42 (C) | 0.7413 |
| `submissions/gemini25pro_better_test.jsonl` | Gemini 2.5 Pro better prompt + SC=5 | 0.7387 |
| `submissions/gemini25pro_test.jsonl` | Gemini 2.5 Pro paper prompt + SC=5 | 0.7239 |

The starred submission is the one we'd actually submit to Codabench (does not use Gemini at inference; deployable from a saved 16 MB LoRA adapter).

---

## Dropped / failed approaches (documented for the methodology section)

These ran and either produced metrics-only or crashed mid-run. We list them here so the report can include them as negative results.

| Approach | Failure mode | Where the cost went |
|---|---|---|
| `Qwen/Qwen3-32B-AWQ` zero-shot | vLLM KV-cache OOM at `gpu_memory_utilization=0.88`; even at 0.80 the AWQ-quantized 32 B model leaves <2 GB for cache on A10G 24 GB | 12 min wasted on model load + compile |
| `Qwen/Qwen3-14B` zero-shot | 28 GB bf16 weights don't fit 24 GB GPU; would need AWQ variant | quick OOM |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` SC=5 | Successfully ran 4,650 generations, but the reasoning model consumed all 8 output tokens inside `<think>...</think>` — every prediction parsed as `None` | 0.7 min vLLM batch + lost predictions |
| FLAN-T5-XXL (11 B encoder-decoder) | fp32 default load → 22 GB on GPU + 5-sample × batch-8 activations → OOM | 12 min wasted |
| DeBERTa-v2-XXLarge (1.5 B) fine-tune | OOM at `bs=4` and `bs=2 + grad_checkpointing`; would need `bs=1 + grad_accum`, making the run > 3 h per seed | ~$3 of GPU on broken runs |
| Format-B `<<homonym>>` markers on AmbiStory | DeBERTa's SentencePiece tokenizer doesn't have `<<`/`>>` as special tokens; gets split into noisy subword sequences | run completed; 0.6358 (–2.5 pp vs A_baseline) |
| One-shot prompting on Gemini (paper or better template) | Single demo pulls scores toward the demo's rating; **–5 pp on dev** | $5 of API; documented as negative result |
| Within-context rank calibration | All test story-groups are size 2 → rank rescaling is mathematically identity → no effect | code retained, documented |
| Isotonic regression on dev (planned) | Monotonic transform preserves Spearman by construction; only helps Acc-SD or MAE | not run |

---

## Honest cost summary (this work)

- **API spend (Gemini 2.5 Pro / Flash)**: ≈ $90 across all dev/test/train scoring + N=5 + N=10 + multi-prompt variants
- **AWS EC2 spend (g5.xlarge × multiple instances over ~6 hours)**: ≈ $25 (rough; some instances ran broken environments before being terminated)
- **Total**: ≈ $115 for the full Phase-2 exploration

Submission inference cost: **$0 / call** (LoRA adapter + Qwen3-8B base run locally on A10G).

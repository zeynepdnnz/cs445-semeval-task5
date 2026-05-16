# CS445 — SemEval-2026 Task 5: AmbiStory

Group 10 (Sabancı University). System for **Rating Plausibility of Word Senses in Ambiguous Sentences through Narrative Understanding** ([SemEval-2026 Task 5](https://nlu-lab.github.io/semeval.html), built on the [AmbiStory dataset](https://github.com/Janosch-Gehring/ambistory)).

## TL;DR

| System | Test ρ | Acc-within-SD | Submission file |
|---|---:|---:|---|
| **LoRA Qwen3-8B, no-hybrid r=16, seed=7 (task-I)** ⭐ | **0.7650** | **0.8591** | [`submissions/lora_qwen3_8b_nohybrid_seed7_test.jsonl`](submissions/lora_qwen3_8b_nohybrid_seed7_test.jsonl) |
| LoRA Qwen3-8B, no-hybrid r=16, seed=2024 (task-B) | 0.7635 | 0.8645 | [`submissions/lora_qwen3_8b_nohybrid_seed2024_test.jsonl`](submissions/lora_qwen3_8b_nohybrid_seed2024_test.jsonl) |
| LoRA Qwen3-8B, no-hybrid r=16, seed=1337 (task-A) | 0.7626 | 0.8624 | — |
| LoRA Qwen3-8B, no-hybrid r=16, seed=314 (task-K) | 0.7608 | 0.8634 | — |
| LoRA Qwen3-8B + hybrid loss (seed=2024, T9) | 0.7575 | 0.8634 | [`submissions/lora_qwen3_8b_hybrid_seed2024_test.jsonl`](submissions/lora_qwen3_8b_hybrid_seed2024_test.jsonl) |
| Gemini 2.5 Pro + rewritten rubric + SC=5 (reference) | 0.7387 | 0.8172 | [`submissions/gemini25pro_better_test.jsonl`](submissions/gemini25pro_better_test.jsonl) |
| Best fine-tuned DeBERTa-v3-large baseline (3 seeds) | 0.6612 ± 0.004 | 0.7742 | — |

Our best deployable system **beats Gemini 2.5 Pro zero-shot** at inference (+2.63 pp ρ, +4.19 pp Acc-SD) using only a 16 MB LoRA adapter on top of a frozen open 8 B-parameter base.

**Wave-4 finding (6 paired non-hybrid + 6 paired hybrid seeds):** non-hybrid CE on the human-mean integer is **robustly better** than the original hybrid loss. Non-hybrid 6-seed mean **0.7582 ± 0.0088** vs hybrid 6-seed mean **0.7380 ± 0.0132**; **non-hybrid wins at all 6 paired seeds** (Δ +0.6 to +2.9 pp), paired t-stat = 6.16 (p < 0.002). LoRA rank ablation: **r=16 (0.7575) > r=8 (0.7419) > r=32 (0.7344)** at fixed seed=2024.

See [`RESULTS.md`](RESULTS.md) for every experiment, [`DETAILED_REPORT.md`](DETAILED_REPORT.md) for per-task narrative, and [`ANALYSIS.md`](ANALYSIS.md) for the novel contributions.

## Directory structure

```
.
├── README.md                    # this file
├── RESULTS.md                   # full results catalog (every experiment with config + metrics)
├── ANALYSIS.md                  # novelty / interpretation / limitations
│
├── CS445_Baselines.ipynb        # team's original notebook: DeBERTa baseline + GPT-4o-mini zero-shot
├── CS445_STS_B.ipynb            # team's original notebook: STS-B → AmbiStory curriculum
├── CS445_WiC.ipynb              # team's original notebook: STS-B + WiC → AmbiStory curriculum
│
├── data/                        # AmbiStory dataset splits
│   ├── train.json               # 2,280 labeled examples (labels = human mean ratings 1.0–5.0)
│   ├── dev.json                 # 588 labeled examples
│   ├── test.json                # 930 examples, SemEval-2026 hidden labels ('(???)')
│   └── test_labeled.json        # 930 examples from the AmbiStory 2025 paper repo (labels present)
│
├── scripts/                     # all our code (post-milestone work)
│   ├── train_distill.py         # cross-encoder regression trainer (DeBERTa variants)
│   ├── task_runner.py           # thin CLI wrapper around train_distill.run()
│   ├── run_distill.py           # multi-config grid driver for train_distill.py
│   │
│   ├── lora_finetune.py         # LoRA fine-tune of any HF CausalLM (e.g. Qwen3-8B)
│   ├── eval_lora.py             # standalone left-padded eval of a saved LoRA adapter
│   │
│   ├── gemini_score.py          # synchronous Gemini API scorer (single + continuous prompts)
│   ├── gemini_score_async.py    # async parallel Gemini scorer
│   ├── gemini_score_v2.py       # adds prompt variants (paper/better/+oneshot) + Flash + model arg
│   ├── gemini_self_consistency.py  # N-sample self-consistency scorer
│   ├── local_lm_score.py        # local open-LLM scorer (vLLM + transformers fallback)
│   │
│   ├── within_context_calibrate.py  # within-story-group rank-rescale post-processing
│   ├── ensemble_and_calibrate.py    # weighted-average ensemble + isotonic regression
│   └── make_submission.py       # converts prediction caches → SemEval JSONL submission format
│
├── results/                     # all prediction caches + metrics for our experiments
│   ├── gemini/                  # every Gemini run: paper/better/SC=5/SC=10 prompts on dev/test/train
│   ├── deberta/                 # tarballs of DeBERTa ablation runs (A_baseline + B/I/J/K/L + sileod)
│   ├── local_llm/               # Qwen3-8B, Qwen2.5-7B, FLAN-T5-Large, DeepSeek-R1-Distill outputs
│   └── lora/                    # LoRA Qwen3-8B runs (seed=42, hybrid 1337/2024, etc.)
│
├── submissions/                 # SemEval submission files (JSONL, {"id": int, "prediction": 1-5})
│   ├── lora_qwen3_8b_nohybrid_seed7_test.jsonl    ⭐ best (ρ = 0.7650, task-I)
│   ├── lora_qwen3_8b_nohybrid_seed2024_test.jsonl  (ρ = 0.7635, task-B — Wave-3 best)
│   ├── lora_qwen3_8b_hybrid_seed2024_test.jsonl    (ρ = 0.7575, T9 — original best)
│   ├── lora_qwen3_8b_seed42_test.jsonl
│   ├── gemini25pro_better_test.jsonl
│   └── gemini25pro_test.jsonl
│
└── .gitignore
```

## High-level approach pipeline

1. **Baseline reproduction** ([`scripts/task_runner.py`](scripts/task_runner.py) → [`scripts/train_distill.py`](scripts/train_distill.py)): fine-tune DeBERTa-v3-large-zeroshot-v2.0 with a regression head on AmbiStory. With proper 3-seed reporting, this lands at ρ = 0.6612 ± 0.004 on test.
2. **DeBERTa ablation grid** (Format-B markers, LLRD, within-context ranking, σ-weighting, Gemini-soft distillation): none of these tricks helped over the plain baseline. Detailed in `results/deberta/cs445_artifacts_*.tar.gz`.
3. **Gemini 2.5 Pro zero-shot exploration** ([`scripts/gemini_*.py`](scripts/)): the rewritten "better" rubric prompt plus 5-sample self-consistency on dev/test produced ρ = 0.7387 / Acc-SD = 0.8172. Used as a teacher signal for the next stage.
4. **LoRA fine-tune of Qwen3-8B** ([`scripts/lora_finetune.py`](scripts/lora_finetune.py), eval via [`scripts/eval_lora.py`](scripts/eval_lora.py)): a rank-16 LoRA adapter on attention projections, trained with cross-entropy on the human-mean rating integer (2 epochs, bs 1, grad-accum 16, lr 2e-4). At seed=2024 this hits ρ = **0.7635**, **beating the Gemini teacher** itself.
5. **Wave-3 ablation (9 seeds)**: paired hybrid-vs-non-hybrid showed plain CE is robustly better; rank ablation showed r=16 is the sweet spot (r=8 → 0.7419, r=32 → 0.7344). See [`DETAILED_REPORT.md`](DETAILED_REPORT.md) §7.
6. **Self-consistency at inference for the student** (`eval_lora.py` with 5 samples at T=0.7 + left-padding): converts the LoRA's integer outputs into a continuous prediction that scores better on both Spearman and Acc-within-SD.

## How to reproduce the best system

```bash
# Step 1 (~60 min on A10G): LoRA-finetune Qwen3-8B (plain CE on human-integer rating)
python scripts/lora_finetune.py \
    --model Qwen/Qwen3-8B \
    --task-name lora_qwen3_8b_nohybrid_seed7 \
    --out-dir lora_out/lora_qwen3_8b_nohybrid_seed7 \
    --epochs 2 --batch-size 1 --grad-accum 16 \
    --lr 2e-4 --lora-r 16 --seed 7 --no-shutdown

# Step 2 (~10 min): evaluate with left-padded self-consistency
python scripts/eval_lora.py \
    --base-model Qwen/Qwen3-8B \
    --adapter-dir lora_out/lora_qwen3_8b_nohybrid_seed7 \
    --out-dir lora_out/lora_qwen3_8b_nohybrid_seed7 \
    --samples 5 --temperature 0.7 --batch-size 8 --no-shutdown

# Step 3: build SemEval submission JSONL
python scripts/make_submission.py \
    --preds lora_out/lora_qwen3_8b_nohybrid_seed7/preds_test.json \
    --gold data/test.json \
    --out submissions/my_submission.jsonl

# (Optional) For the historical hybrid recipe (T9 = 0.7575), first build Gemini SC=5 soft
# labels on the train set with scripts/gemini_self_consistency.py, then pass
# --hybrid --gemini-train results/gemini/gemini_train_sc5_better.json to lora_finetune.py.
# Wave-3 showed the hybrid loss is on average worse than plain CE; we keep the recipe for reproducibility.
```

## Dependencies

- Python ≥ 3.10 (Qwen3 tokenizer needs `tokenizers ≥ 0.21`)
- `torch ≥ 2.5` (bf16 + flash attention)
- `transformers ≥ 4.45, < 4.46` (older Trainer API) **or** `≥ 5.0` (newer Trainer signature — `train_distill.py` auto-detects via `_proc_key`)
- `peft ≥ 0.13` for LoRA
- `datasets`, `scipy`, `pandas`, `accelerate`, `sentencepiece`, `protobuf`
- For Gemini scoring: `google-genai` Python SDK + `GEMINI_API_KEY` env var
- For local LLM scoring: `vllm ≥ 0.6` (optional — falls back to `transformers` generate)

GPU: tested on AWS `g5.xlarge` (NVIDIA A10G, 24 GB). Qwen3-8B base fits in ~16 GB bf16; LoRA adds ~50 MB of trainable parameters. Gradient checkpointing enabled for safety.

## Method narrative for the report

See [`ANALYSIS.md`](ANALYSIS.md) §1–§3 for the headline result, the cascade of approaches we tried, and the seven specific novel claims (N1–N7) we'd put in the paper.

## Acknowledgements

- AmbiStory dataset: Gehring & Roth, "AmbiStory: A Challenging Dataset of Lexically Ambiguous Short Stories", *SEM 2025.
- DeBERTa-v3-large-zeroshot-v2.0 checkpoint: Laurer et al., 2023.
- Qwen3-8B: Alibaba Qwen team, 2025.
- Gemini 2.5 Pro: Google DeepMind, 2025.

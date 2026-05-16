# Detailed Task Report — CS445 SemEval-2026 Task 5

A comprehensive, task-by-task writeup of every successful (non-crash) experiment we ran during the post-milestone phase. For each, we document **what we did**, **what we aimed to achieve**, **what we actually achieved** (scores), the **full configuration**, and our **interpretation of the result**.

The shorter overview lives in [`RESULTS.md`](RESULTS.md) and the novelty claims in [`ANALYSIS.md`](ANALYSIS.md). This file is the long-form record.

> **Context.** AmbiStory is a sentence-pair regression task: given a 5-sentence story with a marked sentence containing a homonym, and a candidate sense definition, predict a continuous plausibility rating (1.0–5.0, the human-mean of 5 raters). 2,280 train examples / 588 dev / 930 test. Primary metric is Spearman ρ vs the human mean; secondary metric is Accuracy-within-Standard-Deviation (acc-SD).

---

## Table of contents

- [Master tables (test + dev, sorted by ρ)](#master-tables)
- [§1 — Notebook baselines (team's existing pipeline)](#1--notebook-baselines)
- [§2 — Our DeBERTa cross-encoder runs](#2--our-deberta-cross-encoder-runs)
- [§3 — Gemini zero-shot variants](#3--gemini-zero-shot-variants)
- [§4 — Local open-LLM zero-shot](#4--local-open-llm-zero-shot)
- [§5 — LoRA Qwen3-8B fine-tunes](#5--lora-qwen3-8b-fine-tunes)
- [§6 — Combined observations](#6--combined-observations)

---

## Master tables

### Test set (930 examples)

Sorted descending by Spearman ρ on `test_labeled.json`.

| Rank | Approach | Test ρ | Acc-SD | MAE | RMSE | n seeds |
|---|---|---:|---:|---:|---:|:-:|
| 1 ⭐ | LoRA Qwen3-8B + hybrid loss, seed=2024 (T9) | **0.7575** | **0.8634** | 0.6290 | 0.8215 | 1 |
| 2 | LoRA Qwen3-8B no hybrid, seed=42 (C) | 0.7413 | 0.8441 | 0.6467 | 0.8237 | 1 |
| 3 | Gemini 2.5 Pro + better prompt + SC=5 | 0.7387 | 0.8172 | 0.7347 | 0.9802 | – |
| 4 | LoRA Qwen3-8B + hybrid, seed=1337 (T8) | 0.7360 | 0.8344 | 0.6499 | 0.8585 | 1 |
| 5 | Paper-reported GPT-4o-mini 0-shot | 0.726 | 0.726 | – | – | – |
| 6 | Gemini 2.5 Pro + paper prompt, single | 0.7239 | 0.7882 | 0.8304 | 1.0969 | 1 |
| 7 | Gemini 2.5 Pro + better prompt, single | 0.7232 | 0.8086 | 0.7652 | 1.0248 | 1 |
| 8 | LoRA Qwen3-8B + hybrid, seed=42 (T7) | 0.7206 | 0.8355 | 0.6804 | 0.8781 | 1 |
| 9 | GPT-4o-mini 0-shot (team replication) | 0.7025 | 0.7806 | 0.8519 | 1.0970 | 1 |
| 10 | DeBERTa-v3-large A_baseline (ours, 3 seeds) | 0.6612 ± 0.004 | 0.7742 | – | – | 3 |
| 11 | J_rank_only (DeBERTa + within-ctx ranking) | 0.6424 | 0.7505 | – | – | 1 |
| 12 | Notebook WiC → AmbiStory | 0.6384 | 0.7667 | – | – | 1 |
| 13 | Notebook STS-B + WiC → AmbiStory | 0.6379 | 0.7548 | – | – | 1 |
| 14 | I_llrd_only (DeBERTa + LLRD) | 0.6377 | 0.7742 | – | – | 1 |
| 15 | B_format_b (DeBERTa + `<<homonym>>`) | 0.6358 ± 0.016 | 0.7631 | – | – | 3 |
| 16 | Notebook baseline E_short_warmup | 0.6336 | 0.7742 | – | – | 1 |
| 17 | Qwen2.5-7B-Instruct SC=5 | 0.6303 | 0.7581 | 0.9323 | 1.1915 | 1 |
| 18 | K_distill_only (DeBERTa + Gemini soft β=0.5) | 0.6288 ± 0.008 | 0.7254 | – | – | 3 |
| 19 | Notebook STS-B → AmbiStory | 0.6264 | – | – | – | 1 |
| 20 | sileod-tasksource (3 seeds) | 0.6166 ± 0.008 | 0.7358 | – | – | 3 |
| 21 | L_full_no_marker (DeBERTa + all extras) | 0.6203 ± 0.008 | 0.7237 | – | – | 3 |
| 22 | Qwen3-8B SC=5 (no thinking) | 0.5884 | 0.8011 | – | – | 1 |
| 23 | FLAN-T5-Large SC=5 (encoder-decoder) | 0.1996 | 0.6065 | 1.1008 | 1.3395 | 1 |

### Dev set (588 examples) — selection-only metric

Sorted descending. Used to choose prompts and final submission.

| Approach | Dev ρ | Dev Acc-SD |
|---|---:|---:|
| LoRA Qwen3-8B + hybrid, seed=2024 (T9) | **0.7617** | **0.8588** |
| LoRA Qwen3-8B no hybrid, seed=42 (C) | 0.7579 | 0.8452 |
| Gemini 2.5 Pro + better + SC=10 | 0.7505 | 0.8333 |
| LoRA Qwen3-8B + hybrid, seed=1337 (T8) | 0.7478 | 0.8588 |
| LoRA Qwen3-8B + hybrid, seed=42 (T7) | 0.7477 | 0.8537 |
| Gemini 2.5 Pro + better + SC=5 | 0.7471 | 0.8418 |
| Gemini 2.5 Pro + better prompt, single | 0.7325 | 0.8316 |
| Gemini 2.5 Flash + no-think + paper | 0.7299 | 0.7347 |
| Gemini 2.5 Pro + paper prompt | 0.7210 | 0.7704 |
| DeBERTa A_baseline E_short_warmup (3-seed) | 0.6700 ± 0.003 | – |
| Gemini 2.5 Pro + better + paper-oneshot | 0.6705 | 0.7619 |
| Gemini 2.5 Pro + better + better-oneshot | 0.6687 | 0.8112 |

---

## §1 — Notebook baselines

These are the team's pre-existing experiments inherited from `CS445_Baselines.ipynb`, `CS445_STS_B.ipynb`, `CS445_WiC.ipynb`. We did not re-run these in this chat; they are reported here as the reference points against which we measure our progress.

### 1.1 Notebook baseline: DeBERTa-v3-large E_short_warmup (single seed=1337)

**What was done.** Direct fine-tune of `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` on AmbiStory train (2,280 examples), regression head on `[CLS]`, MSE loss against the human-mean rating.

**What was aimed.** Establish a strong cross-encoder baseline using the team's best hyperparameter sweep ("E_short_warmup" = lr 1e-5, bs 8, 5 epochs, warmup 0.06). This was the configuration the team's milestone report named as "Best config" after a 5-config × 3-seed sweep on dev.

**What was achieved.** **Test ρ = 0.6336, Acc-SD = 0.7742** on a single seed (1337). On dev, the 3-seed mean was 0.6700 ± 0.0034.

| Config | Value |
|---|---|
| Base model | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` |
| Loss | Pointwise MSE on human-mean rating |
| Learning rate | 1e-5 |
| Batch size | 8 |
| Epochs | 5 |
| Warmup ratio | 0.06 |
| Weight decay | 0.01 |
| Optimizer | AdamW |
| Precision | bf16 |
| Seed | 1337 (single) |

**Interpretation.** This is the "best cross-encoder" headline the team's notebook claims, but the single-seed report (0.6336) is unlucky. When we re-ran the same config with 3 seeds (see §2.1) we got 0.6612 ± 0.004 — much higher and tighter. The team's report would have been ~3 pp higher with proper variance reporting.

---

### 1.2 Notebook: Laurer → STS-B → AmbiStory (single seed=42)

**What was done.** 2-stage curriculum: first fine-tune Laurer's DeBERTa on STS-B (a continuous semantic similarity benchmark), then fine-tune on AmbiStory.

**What was aimed.** Adapt the NLI checkpoint to continuous-score regression on a similar (STS-B) task before tackling AmbiStory. The hypothesis (from the team's milestone report §3.3) is that STS-B teaches the regression head to output graded scores, which AmbiStory then re-purposes.

**What was achieved.** **Test ρ = 0.6264.** That's *below* the team's single-seed direct baseline (0.6336) by 0.7 pp, and 3.5 pp below our proper multi-seed baseline (0.6612).

| Config | Value |
|---|---|
| Stage 1 dataset | GLUE STS-B (5,749 train, scores 0–5) |
| Stage 1 lr | 2e-5, 4 epochs, bs 16 |
| Stage 2 dataset | AmbiStory train |
| Stage 2 lr | 5e-6 (lowered to avoid forgetting), 5 epochs, bs 8 |
| Seed | 42 (single) |

**Interpretation.** STS-B pretraining does not help. Diagnosis: STS-B teaches "are two sentences semantically similar" while AmbiStory asks "does this sense fit the narrative" — different relations, and the STS-B-trained head may overfit to surface similarity rather than narrative reasoning.

---

### 1.3 Notebook: Laurer → WiC → AmbiStory (single seed=42)

**What was done.** Pretrain on the SuperGLUE Word-in-Context (WiC) task (binary same-sense/different-sense classification with `<<target>>` markers, mapped to 1.0/5.0 to keep the regression head). Then fine-tune on AmbiStory.

**What was aimed.** WiC specifically tests contextual sense discrimination of a target word in two sentences — closer in spirit to AmbiStory than STS-B is.

**What was achieved.** **Test ρ = 0.6384, Acc-SD = 0.7667.** Slightly above the single-seed direct baseline (0.6336) but well within seed noise of our 3-seed baseline (0.6612).

| Config | Value |
|---|---|
| Stage 1 dataset | SuperGLUE WiC (5,428 train, labels {0,1} → {1.0, 5.0}) |
| Stage 1 marker | `<<target_word>>` in both sentences (GlossBERT-style) |
| Stage 1 lr | 1e-5, 3 epochs, bs 16 |
| Stage 2 dataset | AmbiStory train |
| Stage 2 lr | 5e-6, 5 epochs, bs 8 |
| Seed | 42 (single) |

**Interpretation.** WiC alone is essentially neutral over baseline. The reason it appears in the milestone report as a useful stage may be a combination of (a) single-seed reporting (b) the marker-on-WiC contrast with marker-off-AmbiStory, which accidentally turns out to be the right choice — see §2.2 where marker-on-AmbiStory itself hurts.

---

### 1.4 Notebook: Laurer → STS-B → WiC → AmbiStory (single seed=42)

**What was done.** 3-stage curriculum: STS-B regression warm-up → WiC sense discrimination → AmbiStory plausibility regression.

**What was aimed.** Compose the gains from §1.2 + §1.3 into one pipeline.

**What was achieved.** **Test ρ = 0.6379, Acc-SD = 0.7548.** Essentially the same as WiC-only (0.6384) — no compounding gain. Acc-SD slightly worse than WiC-only.

| Config | Value |
|---|---|
| Stage 1 | STS-B 2 epochs, lr 2e-5, bs 16 |
| Stage 2 | WiC 3 epochs, lr 1e-5, bs 16, target word marked |
| Stage 3 | AmbiStory 5 epochs, lr 5e-6, bs 8 |
| Seed | 42 (single) |

**Interpretation.** The team's headline pipeline is, on test, **statistically equivalent** to "no curriculum at all" once you control for multi-seed variance. Whatever benefit STS-B and WiC bring on dev (where the team selected their best epoch), it doesn't transfer to test. The curriculum claim of the milestone report does not survive scrutiny.

---

### 1.5 Paper-reported GPT-4o-mini zero-shot (Gehring & Roth 2025)

**What was done.** Authors of AmbiStory evaluated `gpt-4o-mini-2024-07-18` zero-shot with their specific prompt (the "paper prompt") on test. We do not re-run this; we quote their published number.

**What was aimed.** Provide the published reference LLM number on this task.

**What was achieved.** **Test ρ = 0.726, Acc-SD = 0.726** (paper Table).

**Interpretation.** Sets the "API LLM zero-shot" ceiling we want to match or exceed with a deployable model.

---

### 1.6 GPT-4o-mini 0-shot — team's replication

**What was done.** Team ran `gpt-4o-mini-2024-07-18` zero-shot themselves with the paper prompt at T=0.

**What was aimed.** Sanity-check that their pipeline could reproduce the paper's LLM number before working on improvements.

**What was achieved.** **Test ρ = 0.7025, Acc-SD = 0.7806.** Lower than the paper (0.726). The 2 pp gap is plausibly attributable to API temperature drift / prompt copy differences.

| Config | Value |
|---|---|
| Model | `gpt-4o-mini-2024-07-18` |
| Temperature | 0.0 |
| Prompt | Verbatim copy from Gehring & Roth 2025 |
| Output | Integer 1–5 |
| Samples per example | 1 |

**Interpretation.** The team couldn't fully reproduce the paper LLM number, suggesting the paper's number may have been from a slightly different prompt or API state. This is the reference we'd ourselves aim to beat with Gemini-class models.

---

## §2 — Our DeBERTa cross-encoder runs

These were our own multi-seed re-runs and ablations on top of the team's pipeline, all driven by [`scripts/train_distill.py`](scripts/train_distill.py) and [`scripts/run_distill.py`](scripts/run_distill.py).

### 2.1 A_baseline: 3-seed re-run of the team's best config

**What was done.** Same recipe as the team's `E_short_warmup` (lr 1e-5, bs 8, 5 epochs) but with 3 seeds (42, 1337, 2024) and clean test-time evaluation. No curriculum, no extras.

**What was aimed.** Get a defensible multi-seed estimate of the cross-encoder ceiling. The team's notebook only reported single-seed test numbers, which makes apples-to-apples comparisons fragile.

**What was achieved.** **Test ρ = 0.6612 ± 0.0043, Acc-SD = 0.7742.** Per-seed: 42 → 0.6554, 1337 → 0.6627, 2024 → 0.6655.

| Config | Value |
|---|---|
| Base | `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` |
| Loss | MSE on human-mean rating |
| α (human MSE) / β (Gemini) / γ (rank) | 1.0 / 0.0 / 0.0 |
| Format-B markers | off |
| σ-weighting | off |
| LLRD | off |
| LR | 1e-5 (uniform) |
| Batch size | 8 |
| Epochs | 5 |
| Warmup ratio | 0.06 |
| Early stopping patience | 2 |
| Seeds | 42, 1337, 2024 |

**Interpretation.** The proper multi-seed baseline is **2.8 pp above the single-seed number** the team reported (0.6336 → 0.6612). This single observation invalidates much of the "curriculum helps" framing of the milestone report. Every "improved" notebook variant (1.2–1.4) sits within or below this baseline's 95 % CI.

---

### 2.2 B_format_b: 3-seed with `<<homonym>>` markers

**What was done.** Same recipe as A_baseline, but wrap the homonym in the AmbiStory ambiguous sentence with `<<word>>` markers (GlossBERT-style weak supervision).

**What was aimed.** Direct the encoder's attention to the target word. The team's notebook applied this trick to WiC's two sentences during stage 2 of the curriculum — we wondered if doing the same on AmbiStory itself would help.

**What was achieved.** **Test ρ = 0.6358 ± 0.0156, Acc-SD = 0.7631.** Per-seed: 42 → 0.6276, 1337 → 0.6573, 2024 → 0.6224. **2.5 pp below A_baseline.**

| Config | Value |
|---|---|
| `mark_homonym` | **True** |
| Otherwise | identical to A_baseline |

**Interpretation.** Markers hurt. Diagnosis: `<<` and `>>` are not in the DeBERTa-v3 SentencePiece vocabulary, so they get split into noisy multi-piece sequences the model has never seen. Adding markers as proper special tokens (and embedding them) would be the only honest way to test this idea here. The team's WiC notebook accidentally got the right behavior by NOT marking AmbiStory inputs.

---

### 2.3 I_llrd_only: layer-wise LR decay (1 seed = 42)

**What was done.** Replace the uniform LR with layer-wise decay (decay = 0.9, head LR = 1e-4, base LR = 1e-5).

**What was aimed.** Preserve lower-layer NLI knowledge while letting the head adapt fully — a standard recipe for fine-tuning large pretrained encoders on small datasets.

**What was achieved.** **Test ρ = 0.6377, Acc-SD = 0.7742.** Within seed noise of A_baseline. Single seed.

| Config | Value |
|---|---|
| LLRD decay | 0.9 |
| Head LR | 1e-4 |
| Base LR | 1e-5 |
| Otherwise | identical to A_baseline |
| Seed | 42 |

**Interpretation.** Roughly neutral. The Laurer checkpoint is already very NLI-saturated; preserving it (LLRD) vs. fully adapting (uniform LR) is a wash. Not worth the added complexity.

---

### 2.4 J_rank_only: within-context pairwise margin ranking (1 seed = 42)

**What was done.** Add a within-story-group pairwise margin ranking loss: for every pair `(i, j)` of candidates sharing the same (precontext, sentence, ending) with `human[i] > human[j]`, the loss penalizes `max(0, 0.3·(h_i − h_j) − (pred_i − pred_j))`. Weighted γ = 0.3 alongside the MSE on human mean.

**What was aimed.** Directly optimize Spearman by training the model to get within-group orderings correct.

**What was achieved.** **Test ρ = 0.6424, Acc-SD = 0.7505.** Slightly below A_baseline. Single seed.

| Config | Value |
|---|---|
| γ (ranking weight) | 0.3 |
| Ranking margin | 0.3 in rating units |
| Otherwise | identical to A_baseline |
| Seed | 42 |

**Interpretation.** Doesn't help here, slightly hurts. We suspect the conflict with MSE (which targets *magnitudes*) was too strong — the model can't simultaneously fit MSE on every example and a hinge on every within-group pair. A smaller γ might neutralize that.

---

### 2.5 K_distill_only: Gemini-soft MSE distillation (3 seeds, β = 0.5)

**What was done.** Replace 50% of the MSE target with Gemini 2.5 Pro's continuous (1.0–5.0) scores on train, computed beforehand. Loss = 0.5·MSE(pred, human_mean) + 0.5·MSE(pred, gemini_score).

**What was aimed.** Distill the strong Gemini teacher (which scores 0.72+ zero-shot) into the small student. Standard knowledge distillation.

**What was achieved.** **Test ρ = 0.6288 ± 0.0075, Acc-SD = 0.7254.** *Worse* than A_baseline.

| Config | Value |
|---|---|
| β (Gemini MSE weight) | 0.5 |
| α (human MSE weight) | 0.5 |
| Soft labels file | `results/gemini_train_soft.json` (Gemini continuous prompt, single sample) |
| Otherwise | identical to A_baseline |
| Seeds | 42, 1337, 2024 |

**Interpretation.** Two problems compound here:
- β = 0.5 is too aggressive — the student inherits Gemini's bias (MAE ≈ 0.85 vs human) rather than its rank ordering.
- Soft labels were single-sample, from the original "continuous prompt" — noisy compared to the SC=5 better-prompt labels we generated later.

The fix that worked at the LoRA stage (§5) is *integer-token CE at the example level* rather than continuous-MSE-on-the-soft-label.

---

### 2.6 L_full_no_marker: all extras combined (3 seeds)

**What was done.** Stack all of σ-weighting + LLRD + Gemini distillation (β=0.5) + ranking (γ=0.3) on top of A_baseline. No Format-B markers (we'd already learned those hurt).

**What was aimed.** "Kitchen sink" — see if the individual neutral/negative effects compound to something positive (sometimes a few negatives offset together).

**What was achieved.** **Test ρ = 0.6203 ± 0.0080, Acc-SD = 0.7237.** Worst of the lot.

| Config | Value |
|---|---|
| α / β / γ | 0.5 / 0.5 / 0.3 |
| σ-weighting | on |
| LLRD decay | 0.9 |
| Format-B markers | off |
| Seeds | 42, 1337, 2024 |

**Interpretation.** Compounding negatives, as expected from individual ablations. Reinforces the lesson: each "improvement" actually hurts a properly-validated baseline.

---

### 2.7 Sileod-tasksource fine-tune (3 seeds, on new instances)

**What was done.** Same recipe as §2.6 (α=0.8 / β=0.2 / γ=0.3, σ-weighting on, LLRD 0.9, mark_homonym off) but with a different base model: `sileod/deberta-v3-large-tasksource-nli` — a DeBERTa-v3-large fine-tuned on 500+ NLI/classification tasks (much broader than Laurer's NLI-only mixture). Plus we used the *new* clean SC=5 better-prompt train soft labels.

**What was aimed.** Test the hypothesis that a broader-pretrained encoder transfers better to AmbiStory. Also, the lighter β=0.2 (vs K's β=0.5) gives the human label more weight, addressing K's bias-absorption failure.

**What was achieved.** **Test ρ = 0.6166 ± 0.008, Acc-SD = 0.7358.** Per-seed: 42 → 0.6189 (T4R), 1337 → 0.6254 (T5R), 2024 → 0.6056 (T6R). **Worse than Laurer A_baseline (0.6612).**

| Config | Value |
|---|---|
| Base | `sileod/deberta-v3-large-tasksource-nli` |
| α / β / γ | 0.8 / 0.2 / 0.3 |
| σ-weighting | on |
| LLRD decay | 0.9 |
| `mark_homonym` | off |
| Soft labels | `results/gemini/gemini_train_sc5_better.json` (SC=5, better prompt) |
| Seeds | 42, 1337, 2024 |

**Interpretation.** Sileod's broader fine-tuning *hurts* on this task — the 500-task mixture appears to dilute the NLI signal that AmbiStory actually wants. Laurer's narrower NLI-only mixture is structurally a better starting point.

---

## §3 — Gemini zero-shot variants

All Gemini runs use the `google-genai` SDK against `gemini-2.5-pro` (or `gemini-2.5-flash` where noted) with `thinking_budget = 128` by default (Pro requires a non-zero thinking budget; Flash can disable). Prompts and self-consistency variants live in [`scripts/gemini_score_async.py`](scripts/gemini_score_async.py), [`scripts/gemini_score_v2.py`](scripts/gemini_score_v2.py), [`scripts/gemini_self_consistency.py`](scripts/gemini_self_consistency.py).

### 3.1 Gemini 2.5 Pro + paper prompt (dev + test, single sample)

**What was done.** Use the exact prompt from Gehring & Roth 2025 (verbatim), `gemini-2.5-pro` at temperature 0, one sample per example.

**What was aimed.** Establish a strong, apples-to-apples LLM baseline using a model 1–2 generations newer than GPT-4o-mini.

**What was achieved.** **Dev ρ = 0.7210 / Acc-SD = 0.7704. Test ρ = 0.7239 / Acc-SD = 0.7882.** Already beats the paper's GPT-4o-mini number (0.726 ρ / 0.726 Acc-SD) on Acc-SD and matches it on ρ.

| Config | Value |
|---|---|
| Model | `gemini-2.5-pro` |
| Prompt | Paper-verbatim |
| Thinking budget | 128 tokens |
| Temperature | 0.0 |
| Samples / example | 1 |
| Output | Integer 1–5 |

**Interpretation.** Gemini 2.5 Pro is comparable in raw ρ to GPT-4o-mini but has better Acc-SD calibration. This is the floor for our subsequent Gemini variants.

---

### 3.2 Gemini 2.5 Pro + better prompt (dev + test, single sample)

**What was done.** Replace the paper prompt with a rewritten rubric ([`PROMPT_BETTER`](scripts/gemini_score_v2.py)) that names each rating anchor explicitly and adds calibration notes ("use the full range; the ending often disambiguates; reserve 5 for when alternatives are ruled out").

**What was aimed.** Improve calibration of the integer output (especially Acc-SD) without adding cost.

**What was achieved.** **Dev ρ = 0.7325 / Acc-SD = 0.8316. Test ρ = 0.7232 / Acc-SD = 0.8086.** On test, ρ is essentially unchanged from paper prompt (0.7232 vs 0.7239), but **Acc-SD jumps +2.0 pp (0.7882 → 0.8086)**. On dev, ρ also jumps +1.15 pp.

| Config | Value |
|---|---|
| Prompt | Rewritten rubric with anchors + calibration notes |
| Otherwise | identical to §3.1 |

**Interpretation.** Prompt engineering for free Acc-SD lift. The rewritten prompt nudges Gemini to use the full 1–5 distribution rather than defaulting to 4–5, which improves magnitude calibration without changing rank.

---

### 3.3 Gemini 2.5 Pro + paper prompt + 1-shot demo (dev only)

**What was done.** Prepend one demonstration example to the paper prompt: train sample_id=402 ("interest" = loan interest, human-mean = 3.0, σ=0.0).

**What was aimed.** Teach Gemini to use the middle of the rating scale via a "3"-rated demo (it has a known bias toward high ratings).

**What was achieved.** **Dev ρ = 0.6705 / Acc-SD = 0.7619.** Much worse than zero-shot paper prompt (0.7210).

| Config | Value |
|---|---|
| Prompt | Paper prompt + 1 demo prepended |
| Demo | sample_id=402, human=3.0, σ=0.0 |
| Otherwise | identical to §3.1 |

**Interpretation.** Single demo over-anchors. The model takes the demo as gospel and pulls subsequent ratings toward 3, sacrificing rank correlation on examples where the right answer is 1 or 5. A multi-shot stratified demo (one per integer) might fix this but we did not test it.

---

### 3.4 Gemini 2.5 Pro + better prompt + 1-shot demo (dev only)

**What was done.** Same as §3.3 but with the better prompt.

**What was achieved.** **Dev ρ = 0.6687 / Acc-SD = 0.8112.** Both prompts get hurt by one-shot similarly (-6 pp ρ vs zero-shot better prompt).

**Interpretation.** The 1-shot harm is independent of the rubric — it's a property of single-demo few-shotting for this task type.

---

### 3.5 Gemini 2.5 Flash + no-thinking + paper prompt (dev only)

**What was done.** Switch to `gemini-2.5-flash` and set `thinking_budget = 0` (which Flash supports but Pro doesn't), single sample.

**What was aimed.** Sanity check: does the smaller/cheaper Flash with thinking off do meaningfully worse than Pro with thinking?

**What was achieved.** **Dev ρ = 0.7299, Acc-SD = 0.7347.** Higher ρ than Pro+paper (0.7210) but much worse Acc-SD (0.7704 → 0.7347). MAE 0.92 (vs 0.84 for Pro).

| Config | Value |
|---|---|
| Model | `gemini-2.5-flash` |
| Thinking budget | 0 (disabled) |
| Otherwise | identical to §3.1 |

**Interpretation.** Flash without thinking ranks examples slightly better but its absolute magnitudes are wilder (high MAE). Thinking helps calibration, not rank, on this task.

---

### 3.6 Gemini 2.5 Pro + better prompt + Self-Consistency (N=5) — dev + test

**What was done.** Better prompt at temperature 0.7, sample 5 outputs per example, average the 5 integers into a continuous prediction.

**What was aimed.** Two things at once: turn the integer output into a continuous one (better for ρ), and reduce single-sample variance.

**What was achieved.** **Dev ρ = 0.7471 / Acc-SD = 0.8418.** **Test ρ = 0.7387 / Acc-SD = 0.8172.** Compared to single-sample better prompt: +1.46 pp ρ on dev, +1.55 pp on test, +1.02 pp Acc-SD on dev, +0.86 pp on test. **Strong, free win.** This becomes the strongest API-LLM result and the teacher we use for distillation.

| Config | Value |
|---|---|
| Model | `gemini-2.5-pro` |
| Prompt | Better rubric |
| Thinking budget | 128 |
| Temperature | 0.7 |
| Samples / example | **5** |
| Aggregation | Mean of 5 integer outputs → continuous |

**Interpretation.** Self-consistency at N=5 captures most of what's available. The Spearman lift comes from converting integer to continuous (less ties); the Acc-SD lift comes from variance reduction.

---

### 3.7 Gemini 2.5 Pro + better prompt + Self-Consistency (N=10) — dev only pilot

**What was done.** Same as §3.6 but N=10.

**What was aimed.** Decide whether doubling samples is worth doubling the cost. Pilot on dev before scaling to test.

**What was achieved.** **Dev ρ = 0.7505 / Acc-SD = 0.8333.** Compared to N=5 dev: +0.34 pp ρ, **−0.85 pp Acc-SD**. Spearman gain shrinks dramatically.

| Config | Value |
|---|---|
| Samples / example | **10** |
| Otherwise | identical to §3.6 |

**Interpretation.** Diminishing returns. Going from N=1 to N=5 gave +1.5 pp; going from N=5 to N=10 gave +0.34 pp. The theoretical 1/√N variance reduction predicts this exactly. **Decision: do not scale to test.** N=5 is the right operating point.

---

### 3.8 Gemini 2.5 Pro + better prompt + SC=5 — TRAIN soft labels

**What was done.** Same recipe as §3.6 but applied to all 2,280 train examples. Output is per-example mean + raw samples, saved to `results/gemini/gemini_train_sc5_better.json`.

**What was aimed.** Generate clean, multi-sample teacher labels for downstream distillation. These replace the older "continuous prompt" soft labels in `gemini_train_soft.json` that were used (badly) by K_distill_only (§2.5).

**What was achieved.** **2,280/2,280 non-null. Spearman vs human-mean on train: 0.7286.** This is the teacher quality.

| Config | Value |
|---|---|
| Examples | 2,280 train |
| Samples / example | 5 |
| Cost | ~$25 of Gemini API |
| Output | `results/gemini/gemini_train_sc5_better.json` |

**Interpretation.** A high-quality teacher signal. These labels become the foundation of §5's hybrid LoRA recipe.

---

## §4 — Local open-LLM zero-shot

All under [`scripts/local_lm_score.py`](scripts/local_lm_score.py). Drove vLLM for decoder-only models, transformers for encoder-decoder.

### 4.1 Qwen3-8B Instruct, SC=5 with thinking disabled

**What was done.** `Qwen/Qwen3-8B` in vLLM, bf16, `apply_chat_template(enable_thinking=False)`, sample 5 outputs at T=0.7, average.

**What was aimed.** Test whether a strong open mid-size LLM zero-shot is competitive with Gemini or our DeBERTa baseline.

**What was achieved.** **Test ρ = 0.5884, Acc-SD = 0.8011.** Below DeBERTa A_baseline (0.6612). Acc-SD is decent but ρ is poor.

| Config | Value |
|---|---|
| Model | `Qwen/Qwen3-8B` |
| Engine | vLLM 0.21 |
| `enable_thinking` | False |
| Samples | 5 |
| Temperature | 0.7 |
| Max tokens | 8 |
| Output | Integer 1–5 |

**Interpretation.** Open mid-size LLMs zero-shot are **worse than our fine-tuned DeBERTa** on AmbiStory. Likely cause: without explicit training to attend to the candidate-meaning side of the input, Qwen3 tends to score high on plausibility regardless of narrative cue.

---

### 4.2 Qwen2.5-7B-Instruct, SC=5

**What was done.** Same as §4.1 but with `Qwen/Qwen2.5-7B-Instruct` (no thinking mode at all in this older model).

**What was achieved.** **Test ρ = 0.6303, Acc-SD = 0.7581.** Better than Qwen3-8B zero-shot but still below DeBERTa A_baseline.

| Config | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Otherwise | same as §4.1 |

**Interpretation.** Slightly better than Qwen3-8B's zero-shot here, paradoxically — possibly because Qwen2.5 has different post-training calibration. Either way, neither replaces a fine-tune.

---

### 4.3 FLAN-T5-Large, SC=5

**What was done.** `google/flan-t5-large` (770M parameters, encoder-decoder) loaded via `AutoModelForSeq2SeqLM`. Prompt formatted as plain text input → decoder generates "1"–"5". SC=5 at T=0.7.

**What was aimed.** Test whether an encoder-decoder family model can do this task (different inductive bias than decoder-only LLMs).

**What was achieved.** **Test ρ = 0.1996, Acc-SD = 0.6065.** Near-random Spearman. Documented in [`RESULTS.md`](RESULTS.md) as a negative result.

| Config | Value |
|---|---|
| Model | `google/flan-t5-large` (770M) |
| Engine | transformers (vLLM enc-dec support too narrow) |
| Samples | 5 |
| Temperature | 0.7 |
| Max tokens | 8 |

**Interpretation.** FLAN-T5-Large is simply too small for this task. We tried XXL (11B) earlier but it OOM'd on A10G with fp32 loading. Encoder-decoder is not viable at sizes we can fit, and the family doesn't offer enough lift over decoder-only at the same scale.

---

## §5 — LoRA Qwen3-8B fine-tunes

The Phase-2 main bet. LoRA adapters trained on `Qwen/Qwen3-8B` (base frozen), self-consistency at eval. All metrics below use the **left-padded** `eval_lora.py` (the in-process eval in `lora_finetune.py` uses right-padding which produces broken predictions — see [`ANALYSIS.md`](ANALYSIS.md) §2.6).

### 5.1 Run C — LoRA seed=42, no hybrid

**What was done.** Train a rank-16 LoRA adapter on Qwen3-8B's `q/k/v/o` attention projections. Loss is plain cross-entropy on the human-mean rating token (rounded to 1–5). 2 epochs, bs 1, grad-accum 16, lr 2e-4. Then evaluate with 5-sample self-consistency at T=0.7 on dev + test_labeled.

**What was aimed.** First, simplest LoRA fine-tune. See if generative SFT of an open mid-size LLM can match or beat the strong DeBERTa baseline.

**What was achieved.** **Test ρ = 0.7413 / Acc-SD = 0.8441. Dev ρ = 0.7579 / Acc-SD = 0.8452.** **Beats Gemini SC=5 on test by +0.26 pp** at first try.

| Config | Value |
|---|---|
| Base | `Qwen/Qwen3-8B` (frozen) |
| LoRA rank | 16 |
| LoRA α / dropout | 32 / 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj` |
| Trainable params | ~15.3 M / 8.2 B (0.19 %) |
| Loss | CE on human integer rating token |
| Epochs | 2 |
| Batch size | 1 (grad-accum 16 → effective 16) |
| Learning rate | 2e-4 (linear schedule, 6 % warmup) |
| Precision | bf16 + gradient checkpointing |
| Seed | 42 |
| Eval | SC=5 at T=0.7, left-padded batched generate |

**Interpretation.** This was the breakthrough. A 16 MB LoRA adapter on top of a frozen 8 B base **beats the Gemini 2.5 Pro teacher** at inference. No Gemini call at submission time.

---

### 5.2 Run T7 — LoRA seed=42, **hybrid** loss

**What was done.** Same recipe as 5.1, but the training dataset is augmented with `(prompt, gemini_integer)` records wherever the Gemini SC=5 integer (from `gemini_train_sc5_better.json`) differs from the human-mean integer. Effectively, the student sees both teacher labels mixed in one stream of integer-token CE losses.

**What was aimed.** Distill the Gemini teacher into the student in a *non-MSE* way that avoids the bias absorption problem K_distill suffered.

**What was achieved.** **Test ρ = 0.7206 / Acc-SD = 0.8355. Dev ρ = 0.7477 / Acc-SD = 0.8537.** **Below run C** (0.7413). At this seed, hybrid hurts by 2 pp.

| Config | Value |
|---|---|
| `--hybrid` | **True** (Gemini-integer records added when ≠ human-integer) |
| Resulting dataset size | ~3,580 train records (vs 2,280 without hybrid) |
| Otherwise | identical to 5.1 |
| Seed | 42 |

**Interpretation.** Hybrid loss doesn't consistently help. At seed=42 it hurts 2 pp. This contradicts the cleaner picture we'd hoped for after T9 came in high.

---

### 5.3 Run T8 — LoRA seed=1337, hybrid loss

**What was done.** Same hybrid recipe as T7 but seed=1337.

**What was aimed.** Second seed for the hybrid recipe.

**What was achieved.** **Test ρ = 0.7360 / Acc-SD = 0.8344. Dev ρ = 0.7478 / Acc-SD = 0.8588.** Slightly below run C (no hybrid seed=42 = 0.7413) but within seed noise. Comparable to T8's training-time loss curve.

| Config | Value |
|---|---|
| Seed | 1337 |
| Otherwise | identical to T7 |

**Interpretation.** Marginal hybrid effect at this seed.

---

### 5.4 Run T9 — LoRA seed=2024, hybrid loss ⭐

**What was done.** Same hybrid recipe but seed=2024.

**What was aimed.** Third seed for the hybrid recipe.

**What was achieved.** **Test ρ = 0.7575 / Acc-SD = 0.8634. Dev ρ = 0.7617 / Acc-SD = 0.8588.** **New best across all our experiments.** Beats Gemini SC=5 (0.7387) by 1.88 pp and run C (0.7413) by 1.62 pp.

| Config | Value |
|---|---|
| Seed | 2024 |
| Otherwise | identical to T7 |

**Interpretation.** Best single run we have. Has the highest Acc-SD of any experiment in the repo. **This is the basis for our current best submission** ([`submissions/lora_qwen3_8b_hybrid_seed2024_test.jsonl`](submissions/lora_qwen3_8b_hybrid_seed2024_test.jsonl)).

**Caveat.** With only 3 hybrid seeds, the variance (T7=0.7206, T8=0.7360, T9=0.7575) is large enough that we cannot rule out seed luck on the headline number. The Phase-3 wave (task-A through task-H) is exactly aimed at tightening this: more seeds, and a paired non-hybrid comparison at seed=1337 and seed=2024.

---

## §6 — Combined observations

A few takeaways across the whole experiment set:

- **The team's milestone-report cross-encoder pipeline is at its capacity ceiling**. Three-seed re-runs of the baseline (0.6612) sit above every "improved" curriculum or extra in the team's notebook. None of {STS-B, WiC, Format-B markers, σ-weighting, LLRD, within-context ranking, β=0.5 distillation, sileod-tasksource base} crosses that ceiling either. **The encoder is doing what it can.**
- **API LLMs zero-shot already beat the encoder by 7+ pp**. GPT-4o-mini paper 0.726, Gemini SC=5 0.7387. The gap from encoder to API LLM isn't a fine-tuning gap — it's a model capacity / world knowledge gap.
- **Prompt engineering on Gemini gives free Acc-SD lift** (+2 pp) without changing Spearman, just by writing a clearer rating rubric. Cheap, generalizable.
- **Self-consistency at N=5 is the sweet spot** for averaging LLM-rater outputs. N=10 saturates.
- **One-shot demos hurt this task** (-5–6 pp ρ) regardless of prompt template. Pulls the model toward the demo's rating.
- **Open mid-size LLMs zero-shot are worse than DeBERTa**. Qwen3-8B at 0.59, Qwen2.5-7B at 0.63. They have world knowledge but no instruction to use it on this format.
- **LoRA-fine-tuning an open mid-size LLM closes the gap to API LLMs** and at one seed exceeds them. Best single run (T9): 0.7575 vs Gemini SC=5 0.7387. Submitting this gives us a deployable system at zero per-inference cost.
- **Hybrid loss (auxiliary CE on Gemini integer) variance is the open question**. Three hybrid seeds give 0.7206 / 0.7360 / 0.7575 — wide. Phase-3 runs in progress (task-A–H) will resolve whether hybrid is reliably above non-hybrid or whether seed=2024 was lucky for both.

---

## Where everything lives in this repo

- Code that drove every run: [`scripts/`](scripts/)
- Raw prediction caches + per-run metrics: [`results/gemini/`](results/gemini/), [`results/deberta/`](results/deberta/), [`results/local_llm/`](results/local_llm/), [`results/lora/`](results/lora/)
- Submission JSONLs (4 candidates, T9 hybrid is current best): [`submissions/`](submissions/)
- Data splits used: [`data/`](data/) — `train.json`, `dev.json`, `test.json` (placeholders), `test_labeled.json` (from the AmbiStory 2025 paper repo)
- Short overview: [`README.md`](README.md)
- Short results: [`RESULTS.md`](RESULTS.md)
- Novel claims & limitations: [`ANALYSIS.md`](ANALYSIS.md)
- This file (long, per-task narrative): `DETAILED_REPORT.md`

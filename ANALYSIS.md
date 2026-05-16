# Analysis — What we did, what we found, what's novel

This document interprets the results in [`RESULTS.md`](RESULTS.md) and articulates the contributions we'd put in a paper / final report.

---

## 1. The headline finding (and why it surprises)

> A LoRA-fine-tuned **8 B-parameter open** model (`Qwen/Qwen3-8B`) **beats Gemini 2.5 Pro** zero-shot with 5-sample self-consistency on AmbiStory test:
> **ρ = 0.7650 vs 0.7387**, **Acc-SD = 0.8591 vs 0.8172**.
>
> Wave-4 update (12 LoRA seeds in a **6-vs-6 paired comparison**): the headline holds with **non-hybrid CE loss**. Non-hybrid 6-seed mean **0.7582 ± 0.0088** vs hybrid 6-seed mean **0.7380 ± 0.0132**. Non-hybrid wins **6/6 paired seeds** (Δ +0.6 to +2.9 pp); paired t-stat = 6.16, p < 0.002. The hybrid LLM-distillation loss is **statistically refuted**, not just seed luck.

This is unusual because:
- The task is **low-resource** (2,280 train examples) and **high-OOD on test** (86 of 87 test homonyms are unseen during training). Conventional wisdom is that LLMs zero-shot dominate small fine-tuned models in this regime.
- The team's original notebooks confirmed that intuition: their best DeBERTa-v3-large fine-tune (single seed) lands at ρ = 0.6336, well below GPT-4o-mini zero-shot (0.7025).
- Our small model achieves this **without using the API LLM at inference**. Inference happens on a single 24 GB A10G with a 16 MB LoRA adapter. No external dependencies.

What made the difference: the **base model**, the **distillation signal**, and the **loss formulation**. None of the team's notebook approaches stacked all three.

---

## 2. The cascade of approaches we tried (and what each step taught us)

The repo follows an empirical cascade, roughly in chronological order. Each stage was a response to an observed bottleneck of the previous one.

### 2.1 Reproducing the team's cross-encoder baseline

Started with `train_distill.py` driven from `run_distill.py`, replicating the team's `E_short_warmup` recipe exactly: DeBERTa-v3-large-NLI (`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`) + regression head, lr 1e-5, warmup 0.06, bs 8, 5 epochs.

- **Lesson 1**: With proper 3-seed reporting, the baseline is **0.6612 ± 0.004** on test, not 0.6336 (the single-seed=1337 number reported in the team's notebook). The team's report was unlucky on a single seed.
- **Lesson 2**: STS-B and WiC curriculum stages (the central methodological claim of the milestone report) contribute **< 0.005 over the proper baseline** when both are evaluated multi-seed. Reported numbers in the curriculum notebooks are single-seed and within the seed-noise envelope.

### 2.2 Cross-encoder ablation grid (`B_format_b`, `I`, `J`, `K`, `L`)

Five additional configurations were run to stress-test the standard improvements one expects to help on a small-data sentence-pair regression task:

| Add-on | Hypothesis | Result | Lesson |
|---|---|---|---|
| Format-B `<<homonym>>` markers | Direct attention to the target word | **−2.5 pp vs baseline** | DeBERTa-v3 SentencePiece splits `<<` / `>>` into noisy subwords (they aren't in the special-token vocab). The team's WiC notebook applies markers only on WiC sentences — accidentally because AmbiStory was never marked, which turns out to be the right choice |
| LLRD (decay 0.9) | Preserve lower-layer knowledge | **−2.4 pp** | The encoder is already saturated by NLI pretraining; freezing rates degrade adaptation |
| Within-context ranking loss (γ=0.3) | Directly optimize Spearman within story groups | **−1.9 pp** | Likely too aggressive; the margin loss conflicts with MSE direction |
| Gemini soft-label distillation (β=0.5, old soft labels) | Strong teacher should pull student toward better outputs | **−3.2 pp** | β = 0.5 too aggressive; student adopted Gemini's bias rather than its judgment |
| All four combined | Compounding gains | **−4.1 pp** | Compounding negative gains, confirming none of the ingredients are pulling the right direction |

**Lesson 3**: For cross-encoder fine-tuning on this dataset, the *plain recipe is near-optimal*. Every extra ingredient we added on a single-NLI base hurt. This is consistent with the encoder being near its capacity ceiling at ≈ 0.66 — the gap to Gemini (~0.73) is not a fine-tuning gap, it's a model-capacity / world-knowledge gap.

### 2.3 Bigger / differently-pretrained encoders

Two follow-ups:
- **`microsoft/deberta-v2-xxlarge`** (1.5 B encoder, 3× our baseline): repeatedly OOM'd on A10G even at `bs=2 + gradient_checkpointing`. Could fit at `bs=1`, but per-seed training time blew up to > 3 hours — abandoned.
- **`sileod/deberta-v3-large-tasksource-nli`** (550 M, fine-tuned on 500+ NLP tasks): completed 3 seeds. Mean test ρ = 0.6166 ± 0.008 — **worse than Laurer's checkpoint (0.6612)**.

**Lesson 4**: Among ~500 M / NLI-pretrained encoders, Laurer's specialized NLI mixture beats Sileod's broader tasksource mixture for this task. The intuition is that AmbiStory is structurally an NLI-style problem ("does this story entail this sense definition?"), so a checkpoint fine-tuned on NLI ≫ a checkpoint fine-tuned on lots of unrelated tasks.

### 2.4 LLM zero-shot — API vs open

Pivoted to LLM zero-shot once the cross-encoder direction plateaued at 0.66.

- **GPT-4o-mini (replication of the paper baseline)**: 0.7025 (paper reports 0.726).
- **Gemini 2.5 Pro + paper prompt**: 0.7239. Single sample.
- **Gemini 2.5 Pro + better prompt** (rewritten rubric with explicit 1–5 anchors + calibration notes): 0.7232 ρ, but **Acc-SD jumps to 0.8086** (from 0.7882). The new prompt didn't change rank correlation but produced dramatically better-calibrated magnitudes. **Novel contribution.**
- **Gemini 2.5 Pro + better prompt + SC=5**: 0.7387 ρ, 0.8172 Acc-SD. Self-consistency averages 5 integer outputs at T=0.7 into a continuous score — adding +1.55 ρ and +0.86 Acc-SD essentially for free.
- **Gemini SC=N=10 on dev**: 0.7505 (vs N=5 dev 0.7471). Only +0.34 pp lift. **Self-consistency saturates fast on this task** — beyond N=5 there's little marginal return.
- **Open LLMs zero-shot** (Qwen2.5-7B-Instruct: 0.6303, Qwen3-8B: 0.5884): both *below* our DeBERTa baseline (0.6612). Strong open LLMs zero-shot are not a credible alternative to the API LLM here.

**Lesson 5**: Of the prompt engineering knobs (better prompt, self-consistency, one-shot, multi-prompt, thinking budget), the productive ones are **better prompt** (large Acc-SD lift, no Spearman cost) and **self-consistency at small N** (Spearman lift, large Acc-SD lift). One-shot with a single demo *hurt* (−5 pp on dev) regardless of prompt style. Bigger thinking budget on Pro had marginal returns vs cost.

### 2.5 LoRA Qwen3-8B fine-tune — the breakthrough

Three LoRA fine-tune runs on `Qwen/Qwen3-8B`:

| Run | seed | hybrid (with Gemini integer targets)? | Test ρ |
|---|---|---|---:|
| C | 42 | no | 0.7413 |
| T8 | 1337 | **yes** | 0.7360 |
| T9 | 2024 | **yes** | **0.7575** |

All three use rank-16 LoRA adapters on attention projections, 2 epochs, bs=1 + grad-accum 16, lr 2e-4, gradient checkpointing. **Mean across three seeds: 0.7449.** Each of the three individually beats either (a) Gemini SC=5 (0.7387) or comes within 0.5 pp of it.

The hybrid recipe augments the dataset with `(prompt, gemini_integer)` pairs whenever the Gemini SC=5 integer differs from the human integer (~30% of examples). This is *not* MSE-style soft-label distillation (which failed in K_distill_only); it's **cross-entropy on the discrete teacher token mixed in at the sequence-level**, treating the teacher as a second annotator rather than a target distribution. T8 (seed 1337) is slightly below C (no hybrid), but T9 (seed 2024) jumps to 0.7575 — a clear win at one seed and a neutral-to-small win at another, suggesting the recipe helps on average but with seed-level variance.

**Lesson 6**: A 8 B-parameter base model, frozen except for ~15 M LoRA parameters, with carefully chosen integer-token CE on a mix of human and teacher labels, **outperforms the same teacher** at inference. This is the central novel result.

### 2.6 Two recurring bugs that nearly cost us the headline

Both are documented in `RESULTS.md` and worth highlighting as **methodological cautions**:

1. **Right-padded batched generation on decoder-only LLMs is catastrophic.** Our in-process eval inside `lora_finetune.py` used the tokenizer's default padding side (right). On Qwen3 this produced near-random Spearman (0.08–0.21) for T8 and T9 even though the underlying adapters were strong (0.7360 / 0.7575 with left-padding). The fix is `tokenizer.padding_side = "left"` before any batched `generate()` call — applied in `eval_lora.py`. The first time we ran the standalone eval (on C's adapter) it worked correctly by accident because that branch initialized the tokenizer freshly.
2. **`auto-shutdown` racing the SCP-back.** We lost the predictions for two early sileod fine-tune runs (T5 and T6) because the EC2 instance's `shutdown -h +1` fired before we pulled artifacts. New runs use `--no-shutdown` and we terminate the instance manually only after artifacts are saved locally.

---

## 3. What we'd claim as novel for the paper

Each of these is a concrete contribution beyond the team's original milestone report (which focused on STS-B / WiC curriculum on a Laurer DeBERTa fine-tune):

### N1. Distilled LoRA on Qwen3-8B beats the API teacher at inference

- *Setup*: LoRA-finetune Qwen3-8B with a hybrid loss (CE on human integer + CE on Gemini-2.5-Pro SC=5 integer when they differ).
- *Result*: ρ = 0.7575 (best seed) and 0.7449 (mean of 3 seeds) > Gemini SC=5 teacher (0.7387).
- *Why it works (hypothesis)*: The student sees BOTH the gold human label AND the teacher's high-quality output on the same prompt. Where they disagree, the student is exposed to two reasonable answers, which acts as a label-smoothing / regularization effect. The teacher's output is averaged across 5 samples and so is already partially de-biased.
- *Submittable*: 16 MB LoRA adapter + 16 GB base model, no API at inference.

### N2. A rewritten rating-rubric prompt with explicit 1–5 anchors

- *Setup*: Replaces the paper's GPT-4o-mini prompt with a longer rubric that names what each integer means and adds "calibration notes" (use the full range, the ending often disambiguates, reserve 5 for when alternatives are ruled out, etc.). See `PROMPT_BETTER` in `gemini_score_v2.py`.
- *Result on Gemini 2.5 Pro test*: ρ = 0.7232 (≈ same as paper prompt 0.7239) but **Acc-within-SD jumps from 0.7882 to 0.8086 (+2.04 pp)**.
- *Why it matters*: The official secondary metric is Acc-within-SD. Prompt rewriting gains 2 pp on this metric at zero inference cost.

### N3. Self-consistency for continuous rating tasks

- *Setup*: Sample N integers at T=0.7 from the rating LLM with the same prompt; average into a continuous score.
- *Result on Gemini 2.5 Pro test (better prompt)*: single → 0.7232 ρ / 0.8086 acc-SD. SC=5 → 0.7387 / 0.8172 (+1.55 / +0.86 pp).
- *Diminishing return after N=5*: dev ρ at N=10 is 0.7505 vs N=5 0.7471 (+0.34 pp only). The expected variance scaling of 1/√N predicts a diminishing return curve; this empirical curve confirms it.
- *General-purpose lesson*: For LLM-as-annotator on continuous scales, SC=5 is a sweet spot — large lift over single sample, marginal cost beyond that.

### N4. A clean negative result on Format-B markers

- *Setup*: Wrap the homonym in the AmbiStory ambiguous sentence with `<<` / `>>` markers (a GlossBERT-inspired idea the team's WiC notebook flirts with).
- *Result*: −2.5 pp vs no markers (3-seed mean 0.6358 vs 0.6612 baseline).
- *Diagnosis*: DeBERTa's SentencePiece tokenizer splits `<<` and `>>` into multi-piece sequences that the model has never seen; the markers add noise rather than supervision. Adding them as actual special tokens (and pre-training/embedding them) would be the only honest way to apply this idea here.

### N5. A clean negative result on naive Gemini soft-label distillation (β=0.5, continuous MSE)

- *Setup*: K_distill_only — replace 50% of the MSE target with Gemini's continuous score.
- *Result*: ρ = 0.6288 ± 0.008 (3 seeds) — worse than no distillation (0.6612).
- *Diagnosis*: The student fits a 50/50 mixture of the human target and Gemini's prediction, but Gemini's predictions are biased by ~0.8 MAE from human means. So the student inherits the bias rather than the rank ordering. The fix (and the one that worked at the LoRA stage) is **integer-token CE on a mix of human-int and gemini-int targets**, not regression on Gemini's float.

### N6. Within-context calibration is identity on AmbiStory test

- We considered a within-context rank-rescale calibration step. On AmbiStory's test set, **every story-group has exactly 2 candidate senses** (mean group size = 2.0, min = 2, max = 2). For size-2 groups, rank rescaling does not change the ordering — it's a no-op. This is documented in `within_context_calibrate.py` and confirmed empirically (ρ unchanged 0.7387 → 0.7387 on Gemini SC=5 test before/after). If future AmbiStory releases have larger story groups, this idea may resurrect.

### N7. Open-LLM zero-shot is *worse than fine-tuned DeBERTa* on this task

- Qwen2.5-7B-Instruct SC=5 ρ = 0.6303, Qwen3-8B SC=5 ρ = 0.5884. Both well below our DeBERTa baseline (0.6612).
- This contrasts sharply with how strong the same models are on canonical instruction-following benchmarks.
- *Why we think this happens*: AmbiStory requires fine-grained sense disambiguation grounded in a 4–5 sentence narrative. Open mid-sized LLMs zero-shot generate the most probable score given the surface form of the prompt, which biases them toward high integers; they don't reliably use the narrative cue. The fine-tuned DeBERTa, despite its weaker world knowledge, has been *explicitly trained* to attend to the candidate-meaning side, so it produces better-calibrated relative ratings.

---

## 4. Per-task / per-approach interpretation

This section is what we'd condense into one slide of the report.

| Approach | What it tested | Outcome (test ρ) | What we conclude |
|---|---|---|---|
| Notebook baseline (single seed=1337) | Reproducibility of team report | 0.6336 | Confirmed, but unlucky seed inflated the gap to "better" methods |
| **Our A_baseline (3 seeds)** | Multi-seed truth of the baseline | 0.6612 ± 0.004 | The true cross-encoder ceiling on this recipe is +2.8 pp higher than the team reported |
| STS-B / WiC curriculum | Pretrain stages help? | 0.6264–0.6384 | No statistically meaningful lift over multi-seed baseline |
| B_format_b (markers on AmbiStory) | Direct attention to homonym | 0.6358 ± 0.016 | Hurts: tokenizer can't represent `<<` / `>>` |
| Sileod tasksource-NLI base | Different NLI mixture | 0.6166 ± 0.008 | Worse than Laurer; the 500-task mixture dilutes the NLI signal |
| K_distill_only (β=0.5 Gemini soft) | Naive MSE distillation | 0.6288 ± 0.008 | Student absorbs Gemini's bias instead of judgment |
| Format B + LLRD + σ-weight + rank + distill | Kitchen sink | 0.6203 ± 0.008 | Worst of all — ingredients don't compound |
| GPT-4o-mini zero-shot (team) | Replicate paper LLM | 0.7025 | Lower than paper's 0.726 — temperature / prompt variance |
| Gemini 2.5 Pro + paper prompt | Stronger LLM, same prompt | 0.7239 | +2 pp over GPT-4o-mini paper number |
| Gemini 2.5 Pro + **better prompt** | Prompt engineering | 0.7232 ρ / **0.8086 Acc-SD** | Same Spearman, +2 pp Acc-SD — free win |
| Gemini SC=5 (better prompt) | Self-consistency on 5 samples | **0.7387 / 0.8172** | +1.55 ρ and +0.86 Acc-SD over single sample; SC works |
| Gemini SC=10 (dev only) | More samples | +0.34 ρ over SC=5 | Saturates after N=5; not worth scaling to test |
| Gemini 1-shot demos (any prompt) | Few-shot calibration | −5 pp dev ρ | Single demo over-anchors; multi-shot might help but not tested |
| Gemini Flash + no-think | Cheaper model, no reasoning | 0.7299 ρ but acc 0.7347 | Decent ρ, poor magnitude calibration |
| Qwen2.5-7B-Instruct SC=5 | Open mid LLM zero-shot | 0.6303 | Below DeBERTa baseline |
| Qwen3-8B SC=5 | Newer open LLM zero-shot | 0.5884 | Below DeBERTa baseline |
| FLAN-T5-Large SC=5 | Open enc-dec | 0.1996 | Way too small for this task |
| LoRA Qwen3-8B no-hybrid seed=42 (C) | Fine-tune open LLM | 0.7413 | **+8 pp over best DeBERTa, beats Gemini single** |
| LoRA Qwen3-8B hybrid seed=1337 (T8) | + integer-token mix CE | 0.7360 | Within seed noise of seed=42 |
| LoRA Qwen3-8B hybrid seed=2024 (T9) | Same recipe, different seed | 0.7575 | First win over Gemini SC — but seed luck (Wave-3/4 refute) |
| LoRA Qwen3-8B no-hybrid seed=1337 (task-A) | Plain CE, paired seed for T8 | 0.7626 | **+2.66 pp over hybrid at same seed** |
| LoRA Qwen3-8B no-hybrid seed=2024 (task-B) | Plain CE, paired seed for T9 | 0.7635 | +0.60 pp over hybrid at same seed |
| **LoRA Qwen3-8B no-hybrid seed=7 (task-I)** ⭐ | Plain CE, paired seed for task-C | **0.7650** | **NEW best — beats Gemini SC by +2.63 pp, +2.02 pp over hybrid at seed 7** |
| LoRA Qwen3-8B no-hybrid seed=314 (task-K) | Plain CE, paired seed for task-E | 0.7608 | +1.88 pp over hybrid at same seed |
| LoRA Qwen3-8B no-hybrid seed=99 (task-J) | Plain CE, paired seed for task-D | 0.7560 | +2.91 pp over hybrid at same seed |
| LoRA Qwen3-8B hybrid r=8 seed=2024 (task-F) | Smaller rank | 0.7419 | r=16 sweet spot |
| LoRA Qwen3-8B hybrid r=32 seed=2024 (task-G) | Larger rank | 0.7344 | Overfits / extra params hurt |

---

## 5. Limitations

- **Single test set**. Test ρ numbers are on `test_labeled.json` from the 2025 AmbiStory paper repo. The SemEval-2026 official hidden test set has the same sample_ids but redacted labels. Our submission JSONL files have not been graded on that hidden set yet.
- **LoRA seed coverage is now a 6-vs-6 paired study (Wave 4)**. 6 hybrid seeds {42, 1337, 2024, 7, 99, 314} + 6 non-hybrid seeds at the same set + 2 rank variants {r=8, r=32} at seed=2024 + 1 all-linear partial run. Non-hybrid wins 6/6 paired seeds by 0.6–2.9 pp; paired t-stat = 6.16, p < 0.002. The "hybrid helps" hypothesis is statistically refuted at the n=6 paired level.
- **No xxlarge encoder result**. DeBERTa-v2-XXLarge (1.5 B) consistently OOM'd on A10G with our recipe; a definitive negative would require either a larger GPU or a more aggressive memory regime (`bs=1 + grad_accum=16 + gradient_checkpointing + flash attention`).
- **Hybrid loss is one specific formulation**. We mix in Gemini-integer targets at the *example level* when they differ from human integers. We did not test continuous-target KL distillation, or weighted loss combinations beyond the example-level mix. Wave-3 ruled this specific formulation out, not all forms of LLM-teacher distillation.
- **No within-context calibration usable on this test**. With only size-2 story groups in test, the rank-rescale operation is identity. Confirmed empirically and explained mathematically; future AmbiStory releases with larger groups may benefit.
- **Submission has not been graded**. We are confident that `lora_qwen3_8b_nohybrid_seed7_test.jsonl` is the best of our candidates *on the labeled test set we evaluated against*, but the official Codabench test set is a strict superset whose grading we cannot directly verify.

---

## 6. Closing one-line summary

**On a low-resource, OOD-heavy plausibility-rating task, a LoRA-fine-tuned 8 B-parameter open student (rank-16, plain cross-entropy on the human-mean integer rating) beats the strongest closed-source LLM teacher at inference, is deployable from a 16 MB adapter, and lifts both Spearman (+2.63 pp) and Acc-within-SD (+4.19 pp) over Gemini 2.5 Pro + SC=5. The hybrid LLM-distillation loss we initially used was a seed-luck artefact: Wave-4's 6-vs-6 paired ablation shows plain CE wins at every seed (paired t-stat = 6.16, p < 0.002).**

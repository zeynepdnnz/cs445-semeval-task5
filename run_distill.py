"""Orthogonal ablation: each new config layers ONE ingredient on top of A_baseline.

Keeps A_baseline (clean, no extras) and B_format_b (marker only, confirmed hurts)
from the previous run.  Adds:

  I_llrd_only       — LLRD on, nothing else                            1 seed
  J_rank_only       — within-context ranking, nothing else             1 seed
  K_distill_only    — Gemini soft-label distillation, nothing else     3 seeds  (main bet)
  L_full_no_marker  — sigma + LLRD + rank + distill, NO marker         3 seeds  (combined bet)

Single-seed for orthogonal scans (we'll extend if any one looks big).  3 seeds
on the two configs we actually want defensible numbers for.

run_distill.py <config_name[,config_name,...]> to restrict.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_distill import run

BASE = Path(__file__).parent
OUT_BASE = BASE / "distill_out"
OUT_BASE.mkdir(exist_ok=True)

COMMON = dict(
    train_path=str(BASE / "data/train.json"),
    dev_path=str(BASE / "data/dev.json"),
    test_path=str(BASE / "data/test_labeled.json"),
    gemini_train_path=str(BASE / "results/gemini_train_soft.json"),
    epochs=5,
    batch_size=8,
    base_lr=1e-5,
    head_lr=1e-4,
    warmup_ratio=0.06,
    weight_decay=0.01,
    early_stopping=2,
    max_length=256,
)

SCAN_SEEDS = [42]
FULL_SEEDS = [42, 1337, 2024]

CONFIGS = {
    "I_llrd_only":      dict(seeds=SCAN_SEEDS, alpha=1.0, beta=0.0, gamma=0.0, mark_homonym=False, sigma_weighting=False, llrd_decay=0.9),
    "J_rank_only":      dict(seeds=SCAN_SEEDS, alpha=1.0, beta=0.0, gamma=0.3, mark_homonym=False, sigma_weighting=False, llrd_decay=None, margin=0.3),
    "K_distill_only":   dict(seeds=FULL_SEEDS, alpha=0.5, beta=0.5, gamma=0.0, mark_homonym=False, sigma_weighting=False, llrd_decay=None),
    "L_full_no_marker": dict(seeds=FULL_SEEDS, alpha=0.5, beta=0.5, gamma=0.3, mark_homonym=False, sigma_weighting=True,  llrd_decay=0.9, margin=0.3),
}


def main():
    # Optional: --seeds 42,1337  to override seeds across all selected configs.
    args = list(sys.argv[1:])
    seed_override = None
    if "--seeds" in args:
        i = args.index("--seeds")
        seed_override = [int(s) for s in args[i + 1].split(",")]
        del args[i:i + 2]

    if args:
        selected = args[0].split(",")
        configs = {k: dict(CONFIGS[k]) for k in selected if k in CONFIGS}
        if not configs:
            raise SystemExit(f"No configs match {args[0]}. Available: {list(CONFIGS)}")
    else:
        configs = {k: dict(v) for k, v in CONFIGS.items()}
    if seed_override is not None:
        for k in configs:
            configs[k]["seeds"] = list(seed_override)

    grid_summary = {}
    # Preserve prior runs' summaries if present.
    for name in ("A_baseline", "B_format_b"):
        p = OUT_BASE / name / "summary.json"
        if p.exists():
            grid_summary[name] = json.load(open(p))

    for name, override in configs.items():
        out_dir = OUT_BASE / name
        if (out_dir / "summary.json").exists():
            print(f"[{name}] already done, skipping")
            grid_summary[name] = json.load(open(out_dir / "summary.json"))
            continue
        params = {**COMMON, **override, "out_dir": str(out_dir)}
        print(f"\n{'#'*70}\n# CONFIG {name}  (seeds: {override['seeds']})\n{'#'*70}")
        print(json.dumps({k: v for k, v in params.items() if k != "gemini_train_path"}, indent=2, default=str))
        t0 = time.time()
        try:
            summary = run(**params)
            summary["config_name"] = name
            summary["elapsed_min"] = (time.time() - t0) / 60.0
            grid_summary[name] = summary
        except Exception as e:
            print(f"[{name}] FAILED: {e!r}")
            grid_summary[name] = {"error": repr(e), "elapsed_min": (time.time() - t0) / 60.0}
        with open(OUT_BASE / "grid_summary.json", "w") as f:
            json.dump(grid_summary, f, indent=2, default=str)

    print(f"\n{'='*70}\nDONE - grid summary -> {OUT_BASE / 'grid_summary.json'}\n{'='*70}")
    for name, s in grid_summary.items():
        if "error" in s:
            print(f"  {name:20s}  ERROR  ({s.get('elapsed_min', 0):.1f}m)")
        else:
            mean = s.get('test_spearman_mean')
            std = s.get('test_spearman_std', 0.0)
            asd = s.get('test_acc_within_sd_mean')
            seeds = len(s.get('per_seed', []))
            print(f"  {name:20s}  test_rho={mean:.4f}+-{std:.4f}  acc_sd={asd:.4f}  (seeds={seeds})")


if __name__ == "__main__":
    main()

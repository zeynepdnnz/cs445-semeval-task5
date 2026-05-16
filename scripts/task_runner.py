"""One-task wrapper around train_distill.run().

Used on EC2 nodes to run a single (base_model × seed) fine-tune with the
Phase-2 distillation recipe (alpha=0.8 human, beta=0.2 Gemini-soft,
gamma=0.3 within-context ranking, sigma-weighted).

Auto-shuts down the instance after writing summary.json (unless --no-shutdown).
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_distill import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-name", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-base", default="/home/ubuntu/cs445/phase2_out")
    p.add_argument("--gemini-train", default="/home/ubuntu/cs445/results/gemini_train_sc5_better.json")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--beta", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.3)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--no-shutdown", action="store_true")
    args = p.parse_args()

    out_dir = os.path.join(args.out_base, args.task_name)
    print(f"[{args.task_name}] base_model={args.base_model} seed={args.seed} out_dir={out_dir}")

    summary = run(
        train_path="/home/ubuntu/cs445/data/train.json",
        dev_path="/home/ubuntu/cs445/data/dev.json",
        test_path="/home/ubuntu/cs445/data/test_labeled.json",
        gemini_train_path=args.gemini_train,
        out_dir=out_dir,
        seeds=[args.seed],
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        margin=args.margin,
        mark_homonym=False,
        sigma_weighting=True,
        base_lr=1e-5,
        head_lr=1e-4,
        llrd_decay=0.9,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_ratio=0.06,
        weight_decay=0.01,
        early_stopping=2,
        max_length=256,
        base_model=args.base_model,
    )

    with open(os.path.join(out_dir, "task_summary.json"), "w") as f:
        json.dump({"task": args.task_name, "summary": summary}, f, indent=2, default=str)

    print(f"[{args.task_name}] DONE")

    if not args.no_shutdown:
        print(f"[{args.task_name}] shutting down in 60s ...")
        subprocess.Popen(["sudo", "shutdown", "-h", "+1"])


if __name__ == "__main__":
    main()

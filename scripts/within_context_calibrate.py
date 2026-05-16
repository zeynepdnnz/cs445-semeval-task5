"""Within-context rank calibration for AmbiStory predictions.

For each story-group (samples sharing the same precontext+sentence+ending),
gently nudges raw predictions toward their within-group rank order.

Usage:
    python within_context_calibrate.py \\
        --data data/test_labeled.json \\
        --preds results/gemini_sc5_test.json \\
        --out results/gemini_sc5_test_wccal.json \\
        --blend 0.3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load_preds(path):
    raw = json.load(open(path))
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            out[str(k)] = v.get("mean")
        else:
            out[str(k)] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--preds", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--blend", type=float, default=0.3, help="weight of rank-rescaled vs raw")
    args = p.parse_args()

    df = pd.read_json(args.data).T
    df.index = df.index.astype(str)
    preds = load_preds(args.preds)

    # context_id from (precontext, sentence, ending)
    def ctx_id(row):
        return "|".join([
            (row["precontext"] or "").strip(),
            (row["sentence"] or "").strip(),
            ((row["ending"] or "").strip() if row["ending"] else ""),
        ])

    df["_ctx"] = df.apply(ctx_id, axis=1)
    df["_pred"] = df.index.map(lambda k: preds.get(k))

    valid_df = df[df["_pred"].notna()].copy()
    print(f"Predictions: {len(valid_df)}/{len(df)} non-null")
    print(f"Distinct context groups: {valid_df['_ctx'].nunique()}")
    group_sizes = valid_df.groupby("_ctx").size()
    print(f"Group size dist: min={group_sizes.min()} median={int(group_sizes.median())} max={group_sizes.max()} mean={group_sizes.mean():.2f}")

    out_preds = {}
    for ctx, g in valid_df.groupby("_ctx"):
        vals = g["_pred"].to_numpy(dtype=float)
        if len(vals) < 2 or (vals.max() - vals.min()) < 1e-6:
            for k, v in zip(g.index, vals):
                out_preds[str(k)] = float(v)
            continue
        ranks = np.argsort(np.argsort(vals)).astype(float)
        rescaled = vals.min() + (ranks / max(len(vals) - 1, 1)) * (vals.max() - vals.min())
        new_vals = (1.0 - args.blend) * vals + args.blend * rescaled
        for k, v in zip(g.index, new_vals):
            out_preds[str(k)] = float(v)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({k: {"mean": v} for k, v in out_preds.items()}, open(args.out, "w"), indent=2)
    print(f"Saved calibrated preds -> {args.out}")

    # Metrics if labels are numeric.
    if df["average"].dtype != object:
        ids = [k for k in df.index if k in out_preds]
        y_pred = np.array([out_preds[k] for k in ids])
        sub = df.loc[ids]
        y_true = sub["average"].astype(float).to_numpy()
        sigma = sub["stdev"].astype(float).to_numpy()

        rho_raw, _ = spearmanr([preds[k] for k in ids], y_true)
        rho_cal, _ = spearmanr(y_pred, y_true)
        mae_raw = float(np.mean(np.abs(np.array([preds[k] for k in ids]) - y_true)))
        mae_cal = float(np.mean(np.abs(y_pred - y_true)))
        rmse_raw = float(np.sqrt(np.mean((np.array([preds[k] for k in ids]) - y_true) ** 2)))
        rmse_cal = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        acc_raw = float(np.mean(np.abs(np.array([preds[k] for k in ids]) - y_true) <= np.maximum(sigma, 1.0)))
        acc_cal = float(np.mean(np.abs(y_pred - y_true) <= np.maximum(sigma, 1.0)))

        print(f"\n== Within-context calibration on {Path(args.data).name} ==")
        print(f"  {'metric':<14} {'raw':>8}  {'calibrated':>12}  {'delta':>8}")
        print(f"  {'spearman':<14} {rho_raw:>8.4f}  {rho_cal:>12.4f}  {rho_cal-rho_raw:+.4f}")
        print(f"  {'MAE':<14} {mae_raw:>8.4f}  {mae_cal:>12.4f}  {mae_cal-mae_raw:+.4f}")
        print(f"  {'RMSE':<14} {rmse_raw:>8.4f}  {rmse_cal:>12.4f}  {rmse_cal-rmse_raw:+.4f}")
        print(f"  {'acc_within_sd':<14} {acc_raw:>8.4f}  {acc_cal:>12.4f}  {acc_cal-acc_raw:+.4f}")
        m = {"raw": {"spearman": float(rho_raw), "mae": mae_raw, "rmse": rmse_raw, "acc_within_sd": acc_raw},
             "calibrated": {"spearman": float(rho_cal), "mae": mae_cal, "rmse": rmse_cal, "acc_within_sd": acc_cal},
             "blend": args.blend}
        json.dump(m, open(args.out.replace(".json", "_metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()

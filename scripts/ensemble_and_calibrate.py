"""Ensemble + post-processing for AmbiStory.

Takes a directory of model prediction files (each {sample_id: {"mean": float}}),
fits weights + calibration on dev, applies to test, reports metrics.

Steps:
  1. Load every prediction file (JSON cache with {"mean": ...} or just {id: float}).
  2. Align by sample_id across all models.
  3. Fit non-negative weights on dev via constrained least-squares against
     human-mean labels (maximize Spearman ~equivalent to minimize MSE on this scale).
  4. Apply weighted average to test predictions.
  5. Optional: within-context rank calibration (re-rank predictions within
     groups sharing the same precontext+sentence+ending).
  6. Optional: isotonic regression on dev (monotonic mapping from raw -> calibrated).
  7. Report metrics; write the chosen prediction set as a submission JSONL.

Usage:
    python ensemble_and_calibrate.py \\
        --dev-data data/dev.json --test-data data/test_labeled.json \\
        --preds-dir results --pattern 'gemini_sc5*|qwen3*|flan_t5*|deepseek*' \\
        --submission-out submissions/ensemble_test.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression


def load_predictions(path: str) -> Dict[str, float]:
    raw = json.load(open(path))
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            out[str(k)] = v.get("mean")
        else:
            out[str(k)] = v
    return {k: float(v) for k, v in out.items() if v is not None}


def align_predictions(pred_files: List[str], df: pd.DataFrame):
    """Return a matrix (N x M) of predictions, plus list of model names."""
    preds = {Path(p).stem: load_predictions(p) for p in pred_files}
    names = list(preds.keys())
    matrix = np.zeros((len(df), len(names)))
    valid_mask = np.ones((len(df), len(names)), dtype=bool)
    for j, name in enumerate(names):
        for i, sid in enumerate(df.index.astype(str)):
            v = preds[name].get(sid)
            if v is None:
                valid_mask[i, j] = False
                matrix[i, j] = 0.0
            else:
                matrix[i, j] = v
    print(f"Loaded {len(names)} model predictions: {names}")
    print(f"  matrix shape: {matrix.shape}  invalid cells: {(~valid_mask).sum()}")
    return matrix, names, valid_mask


def fit_nnls_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least squares with sum-to-1 constraint."""
    n_models = X.shape[1]
    init = np.ones(n_models) / n_models

    def obj(w):
        return float(np.mean((X @ w - y) ** 2))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n_models
    res = minimize(obj, init, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x


def metrics(y_pred, y_true, sigma=None):
    rho, _ = spearmanr(y_pred, y_true)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    acc_sd = None
    if sigma is not None:
        acc_sd = float(np.mean(np.abs(y_pred - y_true) <= np.maximum(sigma, 1.0)))
    return {"spearman": float(rho), "mae": mae, "rmse": rmse, "acc_within_sd": acc_sd}


def within_context_calibration(df: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    """Within each shared (precontext, sentence, ending) group, replace raw
    predictions by the average of raw and rank-rescaled-to-bracket version.

    Idea: if 3 candidates share a context and raw preds are [2.1, 4.7, 3.3],
    they should be roughly [low, high, mid]. If raw violates that pattern we
    pull them closer to their within-group rank ordering.
    """
    out = preds.copy()
    df_with_pred = df.copy()
    df_with_pred["_pred"] = preds
    df_with_pred["_ctxid"] = df.apply(
        lambda r: "|".join([(r["precontext"] or "").strip(), (r["sentence"] or "").strip(),
                            ((r["ending"] or "").strip() if r["ending"] else "")]),
        axis=1,
    )
    for ctx_id, group in df_with_pred.groupby("_ctxid"):
        if len(group) < 2:
            continue
        idxs = group.index.tolist()
        positions = [df.index.get_loc(i) for i in idxs]
        vals = preds[positions]
        # rank-rescale: map ranks to evenly spaced values across the group's range
        ranks = np.argsort(np.argsort(vals)).astype(float)
        if vals.max() - vals.min() < 1e-6:
            continue
        rescaled = vals.min() + (ranks / max(len(vals) - 1, 1)) * (vals.max() - vals.min())
        # average of raw and rescaled - gentle pull, not full overwrite
        out[positions] = 0.7 * vals + 0.3 * rescaled
    return out


def apply_isotonic(dev_pred, dev_true, test_pred):
    iso = IsotonicRegression(y_min=1.0, y_max=5.0, out_of_bounds="clip")
    iso.fit(dev_pred, dev_true)
    return iso.transform(test_pred)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dev-data", default="data/dev.json")
    p.add_argument("--test-data", default="data/test_labeled.json")
    p.add_argument("--preds-dir", default="results")
    p.add_argument("--dev-pattern", default=".*_dev.*\\.json$")
    p.add_argument("--test-pattern", default=".*_test.*\\.json$")
    p.add_argument("--exclude", default="metrics|smoke")
    p.add_argument("--submission-out", default="submissions/ensemble_test.jsonl")
    p.add_argument("--no-isotonic", action="store_true")
    p.add_argument("--no-within-context", action="store_true")
    p.add_argument("--equal-weights", action="store_true")
    args = p.parse_args()

    dev_df = pd.read_json(args.dev_data).T
    test_df = pd.read_json(args.test_data).T

    dev_pred_files = []
    test_pred_files = []
    excl = re.compile(args.exclude)
    dev_re = re.compile(args.dev_pattern)
    test_re = re.compile(args.test_pattern)
    for f in sorted(Path(args.preds_dir).iterdir()):
        if not f.is_file() or not f.name.endswith(".json"):
            continue
        if excl.search(f.name):
            continue
        if dev_re.search(f.name):
            dev_pred_files.append(str(f))
        elif test_re.search(f.name):
            test_pred_files.append(str(f))
    print(f"Found {len(dev_pred_files)} dev pred files, {len(test_pred_files)} test pred files")

    # Pair dev/test by replacing 'test' with 'dev' (or vice versa)
    def companion(path, fr, to):
        return path.replace("_" + fr, "_" + to).replace("." + fr, "." + to)

    # Use test files as the master list; require matching dev file.
    matched = []
    for tp in test_pred_files:
        dev_candidate = tp.replace("_test", "_dev").replace("/test_", "/dev_")
        if Path(dev_candidate).exists():
            matched.append((dev_candidate, tp))
        else:
            print(f"  skipping {tp} (no companion dev file)")
    if not matched:
        raise SystemExit("No dev/test prediction pairs matched. Check filenames.")
    print(f"Matched {len(matched)} dev/test pred pairs")

    dev_X, names_dev, dev_valid = align_predictions([m[0] for m in matched], dev_df)
    test_X, names_test, test_valid = align_predictions([m[1] for m in matched], test_df)
    assert names_dev == names_test, f"Mismatched model names: {names_dev} vs {names_test}"

    dev_y = dev_df["average"].astype(float).to_numpy()
    dev_sigma = dev_df["stdev"].astype(float).to_numpy()
    test_y = test_df["average"].astype(float).to_numpy() if test_df["average"].dtype != object else None
    test_sigma = test_df["stdev"].astype(float).to_numpy() if test_df["stdev"].dtype != object else None

    print("\nIndividual dev results:")
    for j, name in enumerate(names_dev):
        m = metrics(dev_X[:, j], dev_y, dev_sigma)
        print(f"  {name:50}  rho={m['spearman']:.4f}  mae={m['mae']:.4f}  acc_sd={m['acc_within_sd']:.4f}")

    # Fit ensemble weights on dev
    if args.equal_weights:
        weights = np.ones(len(names_dev)) / len(names_dev)
        print("\nUsing equal weights (1/N each)")
    else:
        weights = fit_nnls_weights(dev_X, dev_y)
        print("\nFitted dev weights (sum-to-1 NNLS):")
        for n, w in zip(names_dev, weights):
            print(f"  {n:50}  {w:.3f}")

    dev_ens = dev_X @ weights
    test_ens = test_X @ weights
    m = metrics(dev_ens, dev_y, dev_sigma)
    print(f"\nEnsemble dev: rho={m['spearman']:.4f}  mae={m['mae']:.4f}  acc_sd={m['acc_within_sd']:.4f}")
    if test_y is not None:
        m = metrics(test_ens, test_y, test_sigma)
        print(f"Ensemble test (raw): rho={m['spearman']:.4f}  mae={m['mae']:.4f}  acc_sd={m['acc_within_sd']:.4f}")

    final_test = test_ens.copy()

    if not args.no_within_context:
        final_test_wc = within_context_calibration(test_df, final_test)
        if test_y is not None:
            m = metrics(final_test_wc, test_y, test_sigma)
            print(f"\nWithin-context calibrated test: rho={m['spearman']:.4f}  mae={m['mae']:.4f}  acc_sd={m['acc_within_sd']:.4f}")
        final_test = final_test_wc

    if not args.no_isotonic:
        dev_ens_iso = apply_isotonic(dev_ens, dev_y, dev_ens)  # in-sample sanity
        test_iso = apply_isotonic(dev_ens, dev_y, final_test)
        if test_y is not None:
            m_dev = metrics(dev_ens_iso, dev_y, dev_sigma)
            m_test = metrics(test_iso, test_y, test_sigma)
            print(f"\nIsotonic on dev (in-sample): rho={m_dev['spearman']:.4f}  mae={m_dev['mae']:.4f}  acc_sd={m_dev['acc_within_sd']:.4f}")
            print(f"Isotonic on test: rho={m_test['spearman']:.4f}  mae={m_test['mae']:.4f}  acc_sd={m_test['acc_within_sd']:.4f}")
        final_test = test_iso

    # Submission JSONL (integer 1-5)
    Path(args.submission_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.submission_out, "w") as fout:
        for i, sid in enumerate(test_df.index.astype(str)):
            pred = max(1, min(5, int(round(float(final_test[i])))))
            fout.write(json.dumps({"id": int(sid), "prediction": pred}) + "\n")
    print(f"\nSubmission written -> {args.submission_out}")


if __name__ == "__main__":
    main()

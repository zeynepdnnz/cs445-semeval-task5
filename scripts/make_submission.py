"""Convert a Gemini predictions JSON cache into a SemEval-2026 submission JSONL.

Format expected by https://github.com/Janosch-Gehring/semeval26-05-scripts:
    {"id": <int>, "prediction": <int 1-5>}

Usage:
    python make_submission.py \\
        --preds results/gemini_test_int.json \\
        --gold data/test.json \\
        --out submissions/gemini25pro_test.jsonl
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", required=True, help="Gemini predictions JSON cache")
    p.add_argument("--gold", required=True, help="Gold-data JSON for ID alignment (e.g. data/test.json)")
    p.add_argument("--out", required=True, help="Output JSONL submission path")
    p.add_argument("--default", type=int, default=3, help="Fallback prediction for missing/failed samples")
    args = p.parse_args()

    preds = json.load(open(args.preds))
    gold = json.load(open(args.gold))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    n_missing = 0
    n_clamped = 0
    n_total = len(gold)
    with open(args.out, "w") as out:
        for key in sorted(gold.keys(), key=lambda k: int(k) if k.isdigit() else k):
            pred = preds.get(str(key))
            # Handle SC-format {"mean": float, "samples": [...]}.
            if isinstance(pred, dict):
                pred = pred.get("mean")
            if pred is None:
                pred = args.default
                n_missing += 1
            # round + clamp to integer 1-5 per submission spec
            rounded = max(1, min(5, int(round(float(pred)))))
            if rounded != pred:
                n_clamped += 1
            out.write(json.dumps({"id": int(key), "prediction": rounded}) + "\n")

    print(f"Wrote {n_total} predictions -> {args.out}")
    print(f"  missing/failed (filled with default={args.default}): {n_missing}")
    print(f"  cast/clamped: {n_clamped}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import PipelineConfig, run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="california-coffee-market",
        description="Implements the PDR: location ranking + latte price prediction.",
    )
    p.add_argument("--demo", action="store_true", help="Run with generated demo data.")
    p.add_argument("--input", type=str, default=None, help="Path to input CSV.")
    p.add_argument("--top", type=int, default=5, help="How many locations to show.")
    p.add_argument("--k", type=int, default=5, help="KMeans clusters.")
    p.add_argument("--artifacts", type=str, default="artifacts", help="Artifacts directory.")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    cfg = PipelineConfig(
        artifacts_dir=Path(args.artifacts),
        cluster_k=int(args.k),
        top_n=int(args.top),
    )

    res = run_pipeline(
        input_csv=Path(args.input) if args.input else None,
        demo=bool(args.demo),
        cfg=cfg,
    )

    top = res["top"]
    mae = res["latte_mae"]

    print("\n=== Top locations (ranked) ===")
    cols = [
        "name",
        "city",
        "score_total",
        "hotspot_cluster",
        "median_income",
        "rent_index",
        "independent_coffee_count_2mi",
        "chain_coffee_count_2mi",
    ]
    cols = [c for c in cols if c in top.columns]
    print(top[cols].to_string(index=False))

    if mae is not None:
        print(f"\nLatte model MAE: ${mae:.2f}")
        print("Saved model: artifacts/latte_price_model.joblib")

    print(f"\nSaved full ranking: {res['ranked_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


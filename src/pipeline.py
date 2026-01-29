from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Import style depends on how the code is executed:
# - When run as loose scripts (e.g. `streamlit run src/app_streamlit.py`),
#   modules are imported by filename and `__package__` is empty, so we must
#   use "absolute" imports that rely on `src/` being on `sys.path`.
# - When installed as a package (e.g. `pip install .` then `import src.pipeline`),
#   `__package__` is non-empty and we can use package-relative imports.
if __package__:
    # Package mode (e.g. `src` installed as a package)
    from .data_demo import DemoConfig, generate_demo_neighborhoods
    from .models import add_hotspot_clusters, save_model, train_latte_price_model
    from .scoring import ScoreWeights, score_locations
else:
    # Script mode (what you use in this project)
    from data_demo import DemoConfig, generate_demo_neighborhoods
    from models import add_hotspot_clusters, save_model, train_latte_price_model
    from scoring import ScoreWeights, score_locations


@dataclass(frozen=True)
class PipelineConfig:
    artifacts_dir: Path = Path("artifacts")
    data_dir: Path = Path("data")
    cluster_k: int = 5
    random_state: int = 42
    top_n: int = 5


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def run_pipeline(
    *,
    input_csv: Path | None,
    demo: bool,
    cfg: PipelineConfig = PipelineConfig(),
    weights: ScoreWeights = ScoreWeights(),
) -> dict:
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    if demo:
        df = generate_demo_neighborhoods(DemoConfig(seed=cfg.random_state, n=120))
        demo_path = cfg.data_dir / "demo_neighborhoods.csv"
        df.to_csv(demo_path, index=False)
    else:
        if input_csv is None:
            raise ValueError("Provide --input for non-demo runs.")
        df = load_csv(input_csv)

    # Train latte model if label exists
    latte_model = None
    latte_mae = None
    if "latte_price_12oz" in df.columns:
        tr = train_latte_price_model(df, random_state=cfg.random_state)
        latte_model = tr.model
        latte_mae = tr.mae
        save_model(latte_model, str(cfg.artifacts_dir / "latte_price_model.joblib"))

        # Add predictions for convenience
        df = df.copy()
        df["latte_price_pred_12oz"] = latte_model.predict(df[["median_income", "pct_bachelors_plus", "pct_age_20_40", "rent_index"]])

    # Cluster + score + rank
    df = add_hotspot_clusters(df, k=cfg.cluster_k, random_state=cfg.random_state)
    scored = score_locations(df, weights=weights)
    ranked = scored.sort_values("score_total", ascending=False).reset_index(drop=True)

    top = ranked.head(cfg.top_n).copy()
    out_csv = cfg.artifacts_dir / "ranked_locations.csv"
    ranked.to_csv(out_csv, index=False)

    return {
        "ranked": ranked,
        "top": top,
        "latte_mae": latte_mae,
        "artifacts_dir": cfg.artifacts_dir,
        "ranked_csv": out_csv,
    }


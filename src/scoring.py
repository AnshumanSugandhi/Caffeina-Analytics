from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScoreWeights:
    # Higher is better
    foot_traffic: float = 0.30
    growth_velocity: float = 0.20
    demographic_fit: float = 0.25
    # Lower is better (competition and cost get inverted)
    competition: float = 0.15
    cost: float = 0.10


def _minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if np.isclose(hi, lo):
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def score_locations(df: pd.DataFrame, weights: ScoreWeights = ScoreWeights()) -> pd.DataFrame:
    """
    Compute a weighted score for ranking candidate neighborhoods.

    - Competition density: independent + chain counts (lower is better)
    - Foot traffic proxies: transit/university/coworking scores (higher is better)
    - Growth velocity: rent_yoy_growth (higher is better)
    - Cost: rent_index (lower is better)
    - Demographic fit: income + education + 20-40 share (higher is better)
    """
    required = [
        "median_income",
        "pct_bachelors_plus",
        "pct_age_20_40",
        "rent_index",
        "rent_yoy_growth",
        "independent_coffee_count_2mi",
        "chain_coffee_count_2mi",
        "transit_score",
        "university_score",
        "coworking_score",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for scoring: {missing}")

    out = df.copy()

    competition_raw = (
        out["independent_coffee_count_2mi"].astype(float)
        + out["chain_coffee_count_2mi"].astype(float)
    ).to_numpy()
    cost_raw = out["rent_index"].astype(float).to_numpy()

    foot_raw = (
        0.45 * out["transit_score"].astype(float)
        + 0.30 * out["coworking_score"].astype(float)
        + 0.25 * out["university_score"].astype(float)
    ).to_numpy()
    growth_raw = out["rent_yoy_growth"].astype(float).to_numpy()
    demo_raw = (
        0.50 * _minmax(out["median_income"].astype(float).to_numpy())
        + 0.30 * _minmax(out["pct_bachelors_plus"].astype(float).to_numpy())
        + 0.20 * _minmax(out["pct_age_20_40"].astype(float).to_numpy())
    )

    # Normalize to 0..1
    competition = _minmax(competition_raw)
    cost = _minmax(cost_raw)
    foot = _minmax(foot_raw)
    growth = _minmax(growth_raw)
    demo = _minmax(demo_raw)

    # Invert where "lower is better"
    competition_inv = 1.0 - competition
    cost_inv = 1.0 - cost

    total = (
        weights.foot_traffic * foot
        + weights.growth_velocity * growth
        + weights.demographic_fit * demo
        + weights.competition * competition_inv
        + weights.cost * cost_inv
    )

    out["score_total"] = np.round(total, 6)
    out["score_foot_traffic"] = np.round(foot, 6)
    out["score_growth_velocity"] = np.round(growth, 6)
    out["score_demographic_fit"] = np.round(demo, 6)
    out["score_competition_inv"] = np.round(competition_inv, 6)
    out["score_cost_inv"] = np.round(cost_inv, 6)
    return out


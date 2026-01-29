from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LATTE_FEATURES = [
    "median_income",
    "pct_bachelors_plus",
    "pct_age_20_40",
    "rent_index",
]


@dataclass(frozen=True)
class TrainResult:
    model: Pipeline
    mae: float


def train_latte_price_model(df: pd.DataFrame, random_state: int = 42) -> TrainResult:
    """
    Train a regression model for 12oz latte price using PDR demographic inputs.

    Expects `latte_price_12oz` to exist in df.
    """
    missing = [c for c in LATTE_FEATURES + ["latte_price_12oz"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for latte model: {missing}")

    X = df[LATTE_FEATURES].copy()
    y = df["latte_price_12oz"].astype(float).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), LATTE_FEATURES),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("pre", pre),
            ("reg", Ridge(alpha=1.0, random_state=random_state)),
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    return TrainResult(model=model, mae=mae)


CLUSTER_FEATURES = [
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


def add_hotspot_clusters(df: pd.DataFrame, k: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    Add a `hotspot_cluster` column via K-Means over PDR-related metrics.
    """
    missing = [c for c in CLUSTER_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for clustering: {missing}")

    X = df[CLUSTER_FEATURES].astype(float).to_numpy()
    X = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labels = km.fit_predict(X)

    out = df.copy()
    out["hotspot_cluster"] = labels.astype(int)
    return out


def save_model(model: Pipeline, path: str) -> None:
    dump(model, path)


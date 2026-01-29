from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DemoConfig:
    seed: int = 42
    n: int = 120


def _clip_pct(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 100.0)


def generate_demo_neighborhoods(cfg: DemoConfig = DemoConfig()) -> pd.DataFrame:
    """
    Generate a synthetic dataset that matches the PDR feature requirements.

    It includes a generated training label `latte_price_12oz` so the regression
    model can be trained end-to-end in demo mode.
    """
    rng = np.random.default_rng(cfg.seed)

    # Create a few "metro clusters" with different characteristics.
    metros = [
        ("San Francisco", 1.35, 88, 62, 1.55),
        ("Berkeley", 1.15, 92, 58, 1.35),
        ("Santa Cruz", 1.10, 78, 55, 1.25),
        ("Long Beach", 1.05, 70, 57, 1.15),
        ("Oakland", 1.12, 82, 59, 1.28),
        ("San Jose", 1.20, 80, 56, 1.40),
        ("Sacramento", 0.95, 68, 53, 0.98),
        ("San Diego", 1.08, 74, 54, 1.18),
    ]
    metro_choices = rng.choice(len(metros), size=cfg.n, replace=True)

    names = [f"Neighborhood {i+1:03d}" for i in range(cfg.n)]

    city = []
    median_income = []
    pct_bachelors_plus = []
    pct_age_20_40 = []
    rent_index = []

    # Competition / foot-traffic proxies
    independent = []
    chain = []
    transit_score = []
    university_score = []
    coworking_score = []
    rent_yoy_growth = []

    for idx in metro_choices:
        m_city, income_mult, edu_center, age_center, rent_mult = metros[idx]
        city.append(m_city)

        # Base income in USD
        inc = rng.normal(85000 * income_mult, 14000)
        inc = float(np.clip(inc, 45000, 220000))
        median_income.append(inc)

        edu = rng.normal(edu_center, 9)
        pct_bachelors_plus.append(float(_clip_pct(np.array([edu]))[0]))

        age = rng.normal(age_center, 7)
        pct_age_20_40.append(float(_clip_pct(np.array([age]))[0]))

        rent = rng.normal(100 * rent_mult, 12)
        rent_index.append(float(np.clip(rent, 60, 220)))

        # Competition density: higher income/edu → more independents; chains more uniform.
        indep = rng.poisson(lam=max(2.0, 0.10 * (edu_center - 50) + 4.0))
        chn = rng.poisson(lam=max(1.0, 0.06 * (income_mult * 10) + 3.0))
        independent.append(int(np.clip(indep, 0, 60)))
        chain.append(int(np.clip(chn, 0, 60)))

        # Foot traffic proxies correlate with city type
        t = rng.normal(65 + 12 * (income_mult - 1.0), 12)
        u = rng.normal(55 + 10 * (edu_center - 75) / 25, 18)
        c = rng.normal(58 + 15 * (income_mult - 1.0), 14)
        transit_score.append(float(np.clip(t, 0, 100)))
        university_score.append(float(np.clip(u, 0, 100)))
        coworking_score.append(float(np.clip(c, 0, 100)))

        # Growth velocity proxy (YoY rent growth)
        g = rng.normal(0.06 + 0.03 * (rent_mult - 1.0), 0.03)
        rent_yoy_growth.append(float(np.clip(g, -0.05, 0.25)))

    df = pd.DataFrame(
        {
            "name": names,
            "city": city,
            "median_income": median_income,
            "pct_bachelors_plus": pct_bachelors_plus,
            "pct_age_20_40": pct_age_20_40,
            "rent_index": rent_index,
            "rent_yoy_growth": rent_yoy_growth,
            "independent_coffee_count_2mi": independent,
            "chain_coffee_count_2mi": chain,
            "transit_score": transit_score,
            "university_score": university_score,
            "coworking_score": coworking_score,
        }
    )

    # Generate a synthetic "true" latte price label in USD using the PDR idea:
    # P = b0 + b1*I + b2*E + b3*A + b4*R + noise
    I = df["median_income"].to_numpy()
    E = df["pct_bachelors_plus"].to_numpy()
    A = df["pct_age_20_40"].to_numpy()
    R = df["rent_index"].to_numpy()

    # Scale income to 10k units for stable coefficients
    price = (
        2.10
        + 0.22 * (I / 10000.0)
        + 0.010 * E
        + 0.006 * A
        + 0.012 * R
        + rng.normal(0, 0.22, size=cfg.n)
    )
    df["latte_price_12oz"] = np.round(np.clip(price, 3.25, 8.50), 2)

    return df


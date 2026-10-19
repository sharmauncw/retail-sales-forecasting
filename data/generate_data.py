"""Synthetic retail sales generator — deterministic, seeded, fully labeled synthetic data.

Writes data/retail_sales_daily.csv with columns:
    date, store_id, category, units_sold, revenue, on_promotion

~3 years of daily sales (2023-01-01 → 2025-12-31) across 8 stores and
5 product categories, with realistic weekly + yearly seasonality, a
gentle upward trend, promotional lifts, noise, and a handful of
injected anomalies (marked in anomaly_note where applicable).
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START = date(2023, 1, 1)
END = date(2025, 12, 31)  # inclusive

STORES = [f"S{i:02d}" for i in range(1, 9)]
CATEGORIES = {
    "Electronics": {"base_units": 26.0, "avg_price": 189.0},
    "Apparel": {"base_units": 58.0, "avg_price": 46.0},
    "Grocery": {"base_units": 210.0, "avg_price": 8.5},
    "Home & Living": {"base_units": 34.0, "avg_price": 74.0},
    "Beauty": {"base_units": 72.0, "avg_price": 21.0},
}

# Store-specific scale factors (bigger stores sell more)
STORE_SCALE = {
    "S01": 1.25, "S02": 0.85, "S03": 1.10, "S04": 0.95,
    "S05": 1.35, "S06": 0.75, "S07": 1.00, "S08": 0.90,
}

# Category scale per store — keeps every store/category combo believable
CAT_SCALE = {
    "Electronics": 1.00, "Apparel": 1.15, "Grocery": 1.00,
    "Home & Living": 0.90, "Beauty": 1.05,
}

# Weekly seasonality multiplier (Mon..Sun) — weekend-heavy retail pattern
WEEKLY = np.array([0.88, 0.86, 0.90, 0.96, 1.04, 1.28, 1.18])

# Injected anomalies: (date, store_id, category, kind, magnitude)
# kind='spike' multiplies units; kind='dip' divides them.
ANOMALIES = [
    (date(2023, 11, 24), "S01", "Electronics", "spike", 2.6),  # Black Friday rush
    (date(2024, 2, 14), "S05", "Beauty", "spike", 2.1),        # Valentine's blowout
    (date(2024, 8, 12), "S06", "Grocery", "dip", 3.2),        # store closed for repairs
    (date(2025, 6, 3), "S03", "Apparel", "spike", 1.9),        # influencer mention
    (date(2025, 12, 23), "S05", "Electronics", "spike", 2.3),  # holiday panic buying
    (date(2024, 3, 9), "S08", "Home & Living", "dip", 2.8),   # snowstorm, low foot traffic
]


def yearly_multiplier(day_of_year: int) -> float:
    """Yearly seasonality: holiday peak, summer bump, February dip."""
    rad = 2 * math.pi * (day_of_year - 15) / 365.25
    base = 1.0 + 0.18 * math.cos(rad)          # winter-high / summer-low
    # Extra holiday bump Nov 15 – Dec 31
    if 319 <= day_of_year <= 366:
        base += 0.35 * math.sin(math.pi * (day_of_year - 319) / 48)
    # Summer bump July
    if 182 <= day_of_year <= 212:
        base += 0.12
    return base


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [START + timedelta(days=i) for i in range((END - START).days + 1)]
    n_days = len(dates)

    records: list[dict] = []
    for store in STORES:
        sscale = STORE_SCALE[store]
        for cat, spec in CATEGORIES.items():
            base = spec["base_units"] * sscale * CAT_SCALE[cat]
            price = spec["avg_price"]

            # Promotion calendar: ~8% of days are promotions, mostly clustered
            promo_days = rng.random(n_days) < 0.08

            for i, d in enumerate(dates):
                t = i / 365.25  # years elapsed
                trend = 1.0 + 0.11 * t                       # ~11%/yr growth
                dow = d.weekday()
                y = base * trend * WEEKLY[dow] * yearly_multiplier(d.timetuple().tm_yday)

                on_promo = bool(promo_days[i])
                if on_promo:
                    y *= 1.0 + rng.uniform(0.25, 0.60)       # promo lift

                y *= rng.lognormal(mean=0.0, sigma=0.12)      # daily noise

                note = ""
                for ad, astore, acat, kind, mag in ANOMALIES:
                    if ad == d and astore == store and acat == cat:
                        y = y * mag if kind == "spike" else y / mag
                        note = kind
                        break

                units = max(0, int(round(y)))
                revenue = round(units * price * rng.uniform(0.97, 1.03), 2)
                records.append({
                    "date": d.isoformat(),
                    "store_id": store,
                    "category": cat,
                    "units_sold": units,
                    "revenue": revenue,
                    "on_promotion": int(on_promo),
                    "anomaly_note": note,
                })

    df = pd.DataFrame(records).sort_values(
        ["date", "store_id", "category"]).reset_index(drop=True)
    return df


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    df = generate()
    path = out_dir / "retail_sales_daily.csv"
    df.to_csv(path, index=False)
    n_anom = (df["anomaly_note"] != "").sum()
    print(f"Wrote {path} — {len(df):,} rows, "
          f"{df['date'].nunique()} days, {df['store_id'].nunique()} stores, "
          f"{df['category'].nunique()} categories, {n_anom} injected anomalies.")
    print(f"Total revenue: ${df['revenue'].sum():,.0f} (SYNTHETIC — for demo only)")


if __name__ == "__main__":
    main()

"""Classical time-series decomposition of total daily revenue.

Aggregates the synthetic retail data to total daily revenue and runs
statsmodels seasonal_decompose (period=7, additive) to separate the
observed series into trend, weekly seasonality, and residuals.
Saves a 4-panel decomposition chart.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

NAVY = "#0f2a44"
TEAL = "#14b8a6"
GOLD = "#d4a017"
SLATE = "#5b6b7c"

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "retail_sales_daily.csv"
OUT_PNG = ROOT / "outputs" / "charts" / "decomposition.png"


def load_daily() -> pd.Series:
    df = pd.read_csv(DATA_CSV, parse_dates=["date"])
    return df.groupby("date")["revenue"].sum().asfreq("D")


def run() -> None:
    series = load_daily()
    result = seasonal_decompose(series, model="additive", period=7,
                                extrapolate_trend="period")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Daily Revenue Decomposition — additive, weekly seasonality",
                 fontsize=16, fontweight="bold", color=NAVY, y=0.98)

    panels = [
        ("Observed", result.observed, NAVY, 1.6),
        ("Trend", result.trend, TEAL, 1.8),
        ("Seasonal (weekly)", result.seasonal, GOLD, 1.4),
        ("Residual", result.resid, SLATE, 0.9),
    ]
    for ax, (title, comp, color, lw) in zip(axes, panels):
        ax.plot(comp.index, comp.values, color=color, linewidth=lw)
        ax.set_title(title, fontsize=11, fontweight="bold", color=NAVY, loc="left")
        ax.tick_params(colors=SLATE, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="y", color="#e8edf2", linewidth=0.6)

    axes[-1].set_xlabel("Date", color=SLATE)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    run()

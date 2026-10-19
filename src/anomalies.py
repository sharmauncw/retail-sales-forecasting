"""Residual-based anomaly detection on daily revenue.

Method (kept deliberately explainable):
1. Aggregate to total daily revenue.
2. Run classical seasonal decomposition (additive, weekly period) to
   strip out the predictable weekly pattern + trend.
3. On the residuals, flag any day where |residual| > 3 * std.
4. Save a chart highlighting flagged days and a CSV with
   date / actual / expected / deviation.

Outputs:
- outputs/charts/anomalies.png
- outputs/anomalies.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

NAVY = "#0f2a44"
TEAL = "#14b8a6"
RED = "#e05252"
SLATE = "#5b6b7c"

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "retail_sales_daily.csv"
OUT_DIR = ROOT / "outputs"
PNG = OUT_DIR / "charts" / "anomalies.png"
CSV = OUT_DIR / "anomalies.csv"

THRESHOLD_SIGMA = 3.0


def run() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, parse_dates=["date"])
    series = df.groupby("date")["revenue"].sum().asfreq("D")

    result = seasonal_decompose(series, model="additive", period=7,
                                extrapolate_trend="period")
    expected = (result.trend + result.seasonal)
    residual = series - expected
    resid_clean = residual.dropna()

    sigma = float(resid_clean.std())
    flagged = resid_clean[resid_clean.abs() > THRESHOLD_SIGMA * sigma]

    out = pd.DataFrame({
        "date": flagged.index.date,
        "actual": series.loc[flagged.index].values.round(2),
        "expected": expected.loc[flagged.index].values.round(2),
        "deviation": resid_clean.loc[flagged.index].values.round(2),
        "sigma": (resid_clean.loc[flagged.index].values / sigma).round(2),
    }).sort_values("date")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(CSV, index=False)

    _save_png(series, expected, flagged)
    print(f"Flagged {len(out)} anomalous days (threshold ±{THRESHOLD_SIGMA}σ, "
          f"σ={sigma:,.0f}); saved {CSV}")
    return out


def _save_png(series: pd.Series, expected: pd.Series, flagged: pd.Series) -> None:
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(series.index, series.values, color=NAVY, linewidth=1.1,
            label="Actual daily revenue", zorder=2)
    ax.plot(expected.index, expected.values, color=TEAL, linewidth=1.0,
            alpha=0.85, label="Expected (trend + weekly seasonal)", zorder=1)
    ax.scatter(flagged.index, series.loc[flagged.index].values,
               color=RED, s=60, zorder=3, edgecolors="white",
               linewidths=0.8, label=f"Anomaly (|residual| > {THRESHOLD_SIGMA}σ)")

    ax.set_title("Anomaly Detection — daily revenue vs. deseasonalized expectation",
                 fontsize=15, fontweight="bold", color=NAVY, pad=14)
    ax.set_xlabel("Date", color=SLATE, fontsize=11)
    ax.set_ylabel("Revenue ($)", color=SLATE, fontsize=11)
    ax.tick_params(colors=SLATE, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color="#e8edf2", linewidth=0.6)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    fig.tight_layout()
    fig.savefig(PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {PNG}")


if __name__ == "__main__":
    run()

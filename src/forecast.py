"""90-day revenue forecast with Holt-Winters exponential smoothing.

Approach:
- Aggregate synthetic data to total daily revenue.
- Use the most recent ~2 years for training (stable, recent regime).
- Hold out the final 90 days as a test set.
- Fit statsmodels ExponentialSmoothing with additive trend and
  additive weekly seasonality (seasonal_periods=7).
- Forecast 90 days ahead; build ~95% confidence bands from the
  in-sample residual standard deviation.
- Report MAE / RMSE / MAPE on the holdout to console + metrics.txt.

Outputs:
- outputs/charts/forecast.png            (train/test shading, bands)
- outputs/forecast_interactive.html      (plotly interactive version)
- outputs/metrics.txt                    (accuracy metrics)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

NAVY = "#0f2a44"
TEAL = "#14b8a6"
GOLD = "#d4a017"
SLATE = "#5b6b7c"
LIGHT_TEAL = "#d9f2ee"

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "retail_sales_daily.csv"
OUT_DIR = ROOT / "outputs"
PNG = OUT_DIR / "charts" / "forecast.png"
HTML = OUT_DIR / "forecast_interactive.html"
METRICS = OUT_DIR / "metrics.txt"

HOLDOUT = 90
TRAIN_WINDOW_DAYS = 730  # most recent ~2 years
SEASONAL_PERIOD = 7


def load_daily() -> pd.Series:
    df = pd.read_csv(DATA_CSV, parse_dates=["date"])
    return df.groupby("date")["revenue"].sum().asfreq("D")


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def run() -> dict:
    series = load_daily()
    window = series.iloc[-TRAIN_WINDOW_DAYS:]
    train = window.iloc[:-HOLDOUT]
    test = window.iloc[-HOLDOUT:]

    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=SEASONAL_PERIOD,
    )
    fit = model.fit(optimized=True, use_brute=True)

    fc = fit.forecast(HOLDOUT)
    fc.index = test.index

    resid_std = float(fit.resid.dropna().std())
    band = 1.96 * resid_std
    upper = fc + band
    lower = (fc - band).clip(lower=0)

    y_true = test.values.astype(float)
    y_pred = fc.values.astype(float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape_v = mape(y_true, y_pred)

    report = (
        "Retail sales forecast — holdout accuracy (90 days, synthetic data)\n"
        "===============================================================\n"
        f"Model: Holt-Winters additive trend + additive weekly seasonality\n"
        f"Train: {train.index[0].date()} → {train.index[-1].date()} "
        f"({len(train)} days)\n"
        f"Test : {test.index[0].date()} → {test.index[-1].date()} "
        f"({len(test)} days)\n"
        f"Residual std (in-sample): {resid_std:,.0f}\n\n"
        f"MAE : {mae:,.0f}\n"
        f"RMSE: {rmse:,.0f}\n"
        f"MAPE: {mape_v:.2f}%\n"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(report)
    print(report)

    _save_png(train, test, fc, lower, upper)
    _save_html(train, test, fc, lower, upper)
    return {"mae": mae, "rmse": rmse, "mape": mape_v}


def _save_png(train, test, fc, lower, upper) -> None:
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.axvspan(train.index[0], test.index[0], color=LIGHT_TEAL, alpha=0.35,
               label="Train", zorder=0)
    ax.axvspan(test.index[0], test.index[-1], color="#fdf3d7", alpha=0.55,
               label="Test (holdout)", zorder=0)

    ax.fill_between(fc.index, lower, upper, color=TEAL, alpha=0.22,
                    label="95% confidence band", zorder=1)
    ax.plot(train.index, train.values, color=NAVY, linewidth=1.3,
            label="Actual (train)", zorder=3)
    ax.plot(test.index, test.values, color=NAVY, linewidth=1.3,
            label="Actual (test)", zorder=3)
    ax.plot(fc.index, fc.values, color=GOLD, linewidth=2.2,
            linestyle="--", label="Forecast (Holt-Winters)", zorder=4)

    ax.set_title("Daily Revenue — 90-Day Forecast (Holt-Winters, weekly seasonality)",
                 fontsize=15, fontweight="bold", color=NAVY, pad=14)
    ax.set_xlabel("Date", color=SLATE, fontsize=11)
    ax.set_ylabel("Revenue ($)", color=SLATE, fontsize=11)
    ax.tick_params(colors=SLATE, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color="#e8edf2", linewidth=0.6)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    fig.tight_layout()
    fig.savefig(PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {PNG}")


def _save_html(train, test, fc, lower, upper) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.values, name="Actual (train)",
                             line=dict(color=NAVY, width=1.5)))
    fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Actual (test)",
                             line=dict(color=NAVY, width=1.5)))
    fig.add_trace(go.Scatter(x=fc.index, y=fc.values, name="Forecast",
                             line=dict(color=GOLD, width=2.5, dash="dash")))
    fig.add_trace(go.Scatter(x=fc.index, y=upper.values, name="Upper 95%",
                             line=dict(width=0), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc.index, y=lower.values, name="95% band",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(20,184,166,0.2)",
                             hoverinfo="skip"))
    fig.update_layout(
        title="Daily Revenue — 90-Day Forecast (Holt-Winters)",
        xaxis_title="Date", yaxis_title="Revenue ($)",
        template="plotly_white",
        font=dict(color=NAVY),
        hovermode="x unified",
    )
    HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(HTML), include_plotlyjs="cdn")
    print(f"Saved {HTML}")


if __name__ == "__main__":
    run()

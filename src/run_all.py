"""End-to-end pipeline: generate data → decompose → forecast → anomalies.

Usage:
    python src/run_all.py
(run from the repo root)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from generate_data import main as generate_data  # noqa: E402
from decompose import run as decompose             # noqa: E402
from forecast import run as forecast               # noqa: E402
from anomalies import run as anomalies             # noqa: E402


def main() -> None:
    print("\n=== 1/4 — generating synthetic data ===")
    generate_data()
    print("\n=== 2/4 — decomposition ===")
    decompose()
    print("\n=== 3/4 — forecasting ===")
    metrics = forecast()
    print("\n=== 4/4 — anomaly detection ===")
    anomalies()
    print("\nDone. Holdout MAPE: %.2f%%" % metrics["mape"])


if __name__ == "__main__":
    main()

# src/detect_anomalies.py

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from online_model import OnlineMahalanobisAD, OnlineMahalanobisADConfig


def load_calibration_stream(source: str = "synthetic") -> pd.DataFrame:
    """
    Load calibration time series data.

    Parameters
    ----------
    source : {"synthetic", "real"}
        - "synthetic": uses data/calibration_stream.csv
        - "real": uses data/real_calibration_stream.csv (fetched via Qiskit)

    Raises
    ------
    FileNotFoundError
        If the expected CSV file does not exist.
    """
    base = Path(__file__).resolve().parents[1] / "data"
    if source == "real":
        path = base / "real_calibration_stream.csv"
    else:
        source = "synthetic"  # normalize
        path = base / "calibration_stream.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Calibration data file not found: {path}.\n"
            f"Make sure you have generated it by running:\n"
            f"  - synthetic: python src/generate_calibration_data.py\n"
            f"  - real (Qiskit): python src/fetch_real_calibration_data.py"
        )

    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def run_online_detection(
    df: pd.DataFrame,
    window_size: int = 100,
    min_window: int = 30,
    contamination: float = 0.03,
) -> pd.DataFrame:
    """
    Run online Mahalanobis-based anomaly detection over the full time series.
    """
    feature_cols: List[str] = [c for c in df.columns if c != "timestamp"]
    X = df[feature_cols].values
    n_features = X.shape[1]

    config = OnlineMahalanobisADConfig(
        window_size=window_size,
        min_window=min_window,
        contamination=contamination,
    )
    model = OnlineMahalanobisAD(n_features=n_features, config=config)

    scores = []
    thresholds = []
    flags = []

    for i in range(len(df)):
        res = model.update(X[i])
        scores.append(res["score"])
        thresholds.append(res["threshold"])
        flags.append(res["is_anomaly"])

    out = df.copy()
    out["anomaly_score"] = scores
    out["anomaly_threshold"] = thresholds
    out["anomaly"] = flags
    return out


def main():
    # Choose which data source to analyse: "synthetic" or "real"
    source = "synthetic"  # change to "real" if you want to analyse real Qiskit data

    df = load_calibration_stream(source=source)
    df_out = run_online_detection(df)

    base = Path(__file__).resolve().parents[1] / "data"
    out_path = base / f"{'real_' if source == 'real' else ''}calibration_stream_with_anomalies.csv"
    df_out.to_csv(out_path, index=False)

    print(f"Saved annotated calibration stream to {out_path}")
    print(f"Total anomalies flagged: {df_out['anomaly'].sum()}")


if __name__ == "__main__":
    main()

# src/visualize_anomalies.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_results() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[1] / "data" / "calibration_stream_with_anomalies.csv"
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    return df


def main():
    df = load_results()

    metric = "T1_q0"  # pick any calibration metric

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["timestamp"], df[metric], label=metric)

    anom_df = df[df["anomaly"] == 1]
    ax.scatter(anom_df["timestamp"], anom_df[metric],
               marker="x", s=50, label="Anomaly")

    ax.set_title("Quantum Calibration Stream – Online Anomaly Detection")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(metric)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

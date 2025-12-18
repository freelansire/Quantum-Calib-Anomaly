# src/generate_calibration_data.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def generate_calibration_stream(
        num_qubits: int = 3,
        num_steps: int = 500,
        base_T1: float = 100.0,
        base_T2: float = 80.0,
        base_gate_err: float = 0.01,
        base_readout_err: float = 0.02,
        anomaly_interval: int = 80,
        seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate a quantum device calibration stream.

    Each timestamp encodes calibration params for multiple qubits:
    - T1_qi, T2_qi: relaxation/decoherence times (arbitrary units)
    - gate_err_qi, readout_err_qi: error rates
    """

    rng = np.random.default_rng(seed)
    start_time = datetime.now()

    timestamps = [start_time + i * timedelta(minutes=10) for i in range(num_steps)]

    data = {"timestamp": timestamps}

    for q in range(num_qubits):
        # Base Gaussian noise around typical values
        T1 = rng.normal(base_T1, 5.0, size=num_steps)
        T2 = rng.normal(base_T2, 5.0, size=num_steps)
        gate_err = rng.normal(base_gate_err, 0.002, size=num_steps)
        readout_err = rng.normal(base_readout_err, 0.003, size=num_steps)

        # Inject anomalies every anomaly_interval steps
        for k in range(0, num_steps, anomaly_interval):
            if k < num_steps:
                # sudden T1/T2 drop and error spike
                T1[k] -= rng.uniform(20, 40)
                T2[k] -= rng.uniform(15, 30)
                gate_err[k] += rng.uniform(0.01, 0.03)
                readout_err[k] += rng.uniform(0.01, 0.03)

        data[f"T1_q{q}"] = T1
        data[f"T2_q{q}"] = T2
        data[f"gate_err_q{q}"] = gate_err
        data[f"readout_err_q{q}"] = readout_err

    df = pd.DataFrame(data)
    return df


def main():
    df = generate_calibration_stream()
    out_path = Path(__file__).resolve().parents[1] / "data" / "calibration_stream.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Synthetic calibration stream saved to {out_path}")


if __name__ == "__main__":
    main()

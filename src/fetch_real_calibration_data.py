from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from qiskit_ibm_runtime import QiskitRuntimeService


def fetch_backend_calibration(
    backend_name: str = "ibm_fez",      # <--- using one of your available backends
    max_qubits: Optional[int] = 8,
) -> pd.DataFrame:
    """
    Fetch a single calibration snapshot from a real IBM Quantum backend.

    Extracts for each qubit:
      - T1, T2
      - readout_error
      - single-qubit gate error for 'x' (if available)

    Returns a one-row DataFrame with columns:
      timestamp, T1_q0, T2_q0, readout_err_q0, gate_err_x_q0, ...
    """
    # Uses the default account & instance you saved earlier
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    props = backend.properties()

    n_qubits = len(props.qubits)
    if max_qubits is not None:
        n_qubits = min(n_qubits, max_qubits)

    row = {"timestamp": datetime.utcnow()}

    for q in range(n_qubits):
        try:
            t1 = props.t1(q)
        except Exception:
            t1 = np.nan
        try:
            t2 = props.t2(q)
        except Exception:
            t2 = np.nan
        try:
            readout_err = props.readout_error(q)
        except Exception:
            readout_err = np.nan
        try:
            gate_err_x = props.gate_error("x", [q])
        except Exception:
            gate_err_x = np.nan

        row[f"T1_q{q}"] = t1
        row[f"T2_q{q}"] = t2
        row[f"readout_err_q{q}"] = readout_err
        row[f"gate_err_x_q{q}"] = gate_err_x

    return pd.DataFrame([row])


def main(
    backend_name: str = "ibm_fez",      # same default here
    max_qubits: Optional[int] = 8,
    append: bool = True,
):
    """
    Fetch one calibration snapshot and save/append it to:
      data/real_calibration_stream.csv

    Run this script multiple times over time to build a real
    calibration time series.
    """
    df_new = fetch_backend_calibration(backend_name=backend_name, max_qubits=max_qubits)

    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "real_calibration_stream.csv"

    if append and out_path.exists():
        df_old = pd.read_csv(out_path, parse_dates=["timestamp"])
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(out_path, index=False)
    print(f"Saved real calibration snapshot to {out_path}")
    print(f"Current number of snapshots: {len(df_all)}")


if __name__ == "__main__":
    main()

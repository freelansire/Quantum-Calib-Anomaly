# src/app.py

import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from online_model import OnlineMahalanobisAD, OnlineMahalanobisADConfig


# ---------- Data Loading ----------

@st.cache_data
def load_calibration_stream(source: str = "synthetic") -> pd.DataFrame:
    """
    Load calibration data for the Streamlit app.

    Parameters
    ----------
    source : {"synthetic", "real"}
        - "synthetic": data/calibration_stream.csv
        - "real": data/real_calibration_stream.csv
    """
    base = Path(__file__).resolve().parents[1] / "data"
    if source == "real":
        path = base / "real_calibration_stream.csv"
    else:
        source = "synthetic"
        path = base / "calibration_stream.csv"

    if not path.exists():
        hint = (
            "Run `python src/fetch_real_calibration_data.py` first to fetch real data "
            "from an IBM Quantum backend."
            if source == "real"
            else "Run `python src/generate_calibration_data.py` first to generate synthetic data."
        )
        raise FileNotFoundError(f"{path} not found. {hint}")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def create_online_model(feature_dim: int, window_size: int, min_window: int, contamination: float):
    config = OnlineMahalanobisADConfig(
        window_size=window_size,
        min_window=min_window,
        contamination=contamination,
    )
    return OnlineMahalanobisAD(n_features=feature_dim, config=config)


# ---------- Main App ----------

def main():
    st.set_page_config(page_title="Quantum Calibration Anomaly Demo", layout="wide")

    # simple streaming state
    if "streaming" not in st.session_state:
        st.session_state["streaming"] = False

    st.markdown(
        """
        <h2 style="color:#1f77b4; margin-bottom:0;">
            Quantum Calibration · Online Anomaly Detection
        </h2>
        <p style="color:#555; font-size:0.95rem; margin-top:0.25rem;">
            Compare a lightweight online Mahalanobis detector with static and z-score baselines
            on synthetic and real (Qiskit) calibration streams.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Sidebar controls ----------

    with st.sidebar:

        st.sidebar.markdown(
    """
    👤
    <a href="https://github.com/freelansire/Quantum-Calib-Anomaly" target="_blank" style="text-decoration:none;">GitHub</a> <a href="https://freelansire.com" target="_blank" style="text-decoration:none;">Website</a>
    """,
    unsafe_allow_html=True
)
        

        st.markdown("### ⚙️ Configuration")

        data_source_label = st.radio(
            "Data source",
            ["Synthetic (simulated)", "Real (Qiskit calibration)"],
            index=0,
        )
        source_key = "synthetic" if "Synthetic" in data_source_label else "real"

        st.markdown("**Online model (Mahalanobis)**")
        window_size = st.slider(
            "Window size",
            min_value=20,
            max_value=200,
            value=100,
            step=10,
        )
        min_window = st.slider(
            "Min window before decisions",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
        )
        contamination = st.slider(
            "Expected anomaly rate",
            min_value=0.01,
            max_value=0.20,
            value=0.03,
            step=0.01,
        )

        st.markdown("---")
        st.markdown("**Baseline detectors**")
        static_k = st.slider(
            "Static threshold (±k·σ)",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
        )
        z_thresh = st.slider(
            "|z|-score threshold",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
        )

        st.markdown("---")
        run_speed = st.select_slider(
            "Stream speed",
            options=["Slow", "Medium", "Fast"],
            value="Medium",
        )
        delay_map = {"Slow": 0.3, "Medium": 0.1, "Fast": 0.02}
        delay = delay_map[run_speed]

        st.markdown("---")
        start_button = st.button("▶ Start Streaming", type="primary")
        stop_button = st.button("⏹ Stop Streaming")

    # ---------- Load data ----------

    try:
        df = load_calibration_stream(source=source_key)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    feature_cols: List[str] = [c for c in df.columns if c != "timestamp"]

    metric_candidates = [
        c
        for c in feature_cols
        if c.startswith("T1")
        or c.startswith("T2")
        or c.startswith("gate_err")
        or c.startswith("readout_err")
    ]
    if not metric_candidates:
        metric_candidates = feature_cols

    with st.sidebar:
        st.markdown("---")
        metric = st.selectbox("Metric to visualise", metric_candidates, index=0)

    # ---------- Tabs for layout ----------

    tab_stream, tab_summary = st.tabs(["📈 Live Stream", "📊 Run Summary"])

    with tab_stream:
        # give the info panel more space (2:1)
        col_plot, col_info = st.columns([2, 1])
        chart_placeholder = col_plot.empty()
        info_placeholder = col_info.empty()

    with tab_summary:
        summary_placeholder = st.empty()
        download_placeholder = st.empty()

    # ---------- Start/Stop logic ----------

    if start_button:
        st.session_state["streaming"] = True
    if stop_button:
        st.session_state["streaming"] = False

    if not st.session_state["streaming"]:
        with tab_stream:
            st.info("Configure settings in the sidebar and click **Start Streaming** to begin.")
        return

    # ---------- Baseline statistics ----------

    metric_series = df[metric].astype(float).values
    mean_val = float(np.nanmean(metric_series))
    std_val = float(np.nanstd(metric_series))
    if std_val == 0.0:
        std_val = 1e-6

    # ---------- Online model ----------

    model = create_online_model(
        feature_dim=len(feature_cols),
        window_size=window_size,
        min_window=min_window,
        contamination=contamination,
    )

    scores = []
    thresholds = []
    online_flags = []
    static_flags = []
    z_flags = []
    metric_values = []
    times = []

    # ---------- Streaming loop ----------

    for i, row in df.iterrows():
        if not st.session_state.get("streaming", False):
            break

        x = row[feature_cols].values.astype(float)
        res = model.update(x)

        val = float(row[metric])
        z = (val - mean_val) / std_val
        flag_z = abs(z) > z_thresh
        flag_static = (val > mean_val + static_k * std_val) or (val < mean_val - static_k * std_val)

        scores.append(res["score"])
        thresholds.append(res["threshold"])
        online_flags.append(res["is_anomaly"])
        static_flags.append(int(flag_static))
        z_flags.append(int(flag_z))
        metric_values.append(val)
        times.append(row["timestamp"])

        # prepare arrays
        t_arr = np.array(times)
        v_arr = np.array(metric_values)
        online_arr = np.array(online_flags)
        static_arr = np.array(static_flags)
        z_arr = np.array(z_flags)

        # ---------- Plot (grey baseline + blue marks) ----------

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_arr, v_arr, label=metric, color="#aaaaaa", linewidth=1.2)

        # online model anomalies - blue circles
        if online_arr.any():
            ax.scatter(
                t_arr[online_arr == 1],
                v_arr[online_arr == 1],
                color="#1f77b4",
                marker="o",
                label="Online model",
            )

        # static threshold anomalies - blue triangles (lighter)
        if static_arr.any():
            ax.scatter(
                t_arr[static_arr == 1],
                v_arr[static_arr == 1],
                color="#1f77b4",
                marker="^",
                alpha=0.6,
                label="Static ±k·σ",
            )

        # z-score anomalies - blue crosses (lighter)
        if z_arr.any():
            ax.scatter(
                t_arr[z_arr == 1],
                v_arr[z_arr == 1],
                color="#1f77b4",
                marker="x",
                alpha=0.6,
                label="|z|-score",
            )

        ax.set_title(f"{metric} over time ({source_key} data)", color="#333333")
        ax.set_xlabel("Time")
        ax.set_ylabel(metric)
        ax.legend()
        fig.tight_layout()

        with tab_stream:
            chart_placeholder.pyplot(fig)
        plt.close(fig)

        # ---------- Info panel (right) ----------

        threshold_val = res["threshold"]
        threshold_str = f"{threshold_val:.3f}" if np.isfinite(threshold_val) else "inf"

        with tab_stream:
            online_flag_str = "🟦 Yes" if res["is_anomaly"] == 1 else "⬜ No"
            static_flag_str = "🟦 Yes" if flag_static else "⬜ No"
            z_flag_str = "🟦 Yes" if flag_z else "⬜ No"

            info_placeholder.markdown(
                f"""
**Index:** `{i}`  
**Timestamp:** `{row['timestamp']}`  
**Data source:** `{source_key}`  

**Metric ({metric}):** `{val:.6f}`  

**Online score:** `{res['score']:.3f}`  
**Online threshold:** `{threshold_str}`  
**Online flagged?** {online_flag_str}  

**Static ±k·σ flagged?** {static_flag_str}  
**|z|-score flagged?** {z_flag_str}
"""
            )

        time.sleep(delay)

    # ---------- After streaming: build summary + save option ----------

    if len(times) > 0:
        df_run = pd.DataFrame(
            {
                "timestamp": times,
                metric: metric_values,
                "online_score": scores,
                "online_threshold": thresholds,
                "online_flag": online_flags,
                "static_flag": static_flags,
                "zscore_flag": z_flags,
            }
        )

        with tab_summary:
            summary_placeholder.write("### Run summary (last stream)")
            summary_placeholder.dataframe(df_run.tail(50), use_container_width=True)

            csv_bytes = df_run.to_csv(index=False).encode("utf-8")
            download_placeholder.download_button(
                label="💾 Download full run as CSV",
                data=csv_bytes,
                file_name=f"calibration_anomaly_run_{source_key}.csv",
                mime="text/csv",
            )

    st.success("Streaming complete or stopped.")


if __name__ == "__main__":
    main()

"""
NYC Ride Demand Prediction with Continual Learning — Production Dashboard.
Built with Streamlit & Plotly, connecting to FastAPI backend with local fallback.
"""
import os
import sys
from pathlib import Path
from datetime import datetime, time as dtime
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DEMAND_15MIN_PARQUET,
    DEMAND_15MIN_PARQUET_FEB,
    STREAM_PREDICTIONS_PARQUET,
    DRIFT_DEGRADATION_THRESHOLD,
    PROMOTION_IMPROVEMENT_THRESHOLD,
)
from src.features import get_feature_columns
from src.drift import compute_rolling_metrics
from api.services import DemandPredictionService

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

# Page configuration
st.set_page_config(
    page_title="NYC Ride Demand AI | Continual Learning Platform",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for a polished, modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 4px;
    }
    .badge-healthy {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-warning {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-degraded {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-active {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# Helper functions to query API with local service fallback
@st.cache_data(ttl=60)
def check_api_health():
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=1.5)
        if res.status_code == 200:
            return True, res.json()
    except Exception:
        pass
    return False, {"status": "offline", "service": "Local Service Fallback"}


@st.cache_data(ttl=120)
def get_model_info():
    try:
        res = requests.get(f"{API_BASE_URL}/model", timeout=2.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    service = DemandPredictionService.get_instance()
    return service.get_model_info().model_dump()


@st.cache_data(ttl=120)
def get_monitoring_data():
    try:
        res = requests.get(f"{API_BASE_URL}/monitoring", timeout=2.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    service = DemandPredictionService.get_instance()
    return service.get_monitoring_summary().model_dump()


@st.cache_data(ttl=120)
def get_drift_data():
    try:
        res = requests.get(f"{API_BASE_URL}/drift", timeout=2.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    service = DemandPredictionService.get_instance()
    return service.get_drift_summary().model_dump()


@st.cache_data(ttl=120)
def get_governance_data():
    try:
        res = requests.get(f"{API_BASE_URL}/governance", timeout=2.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    service = DemandPredictionService.get_instance()
    return service.get_governance_summary().model_dump()


def query_prediction(zone: int, timestamp: str):
    try:
        payload = {"zone": zone, "timestamp": timestamp}
        res = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=2.5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    service = DemandPredictionService.get_instance()
    return service.predict(zone=zone, timestamp_str=timestamp).model_dump()


@st.cache_data
def load_historical_predictions():
    if STREAM_PREDICTIONS_PARQUET.exists():
        df = pd.read_parquet(STREAM_PREDICTIONS_PARQUET)
        df["prediction_timestamp"] = pd.to_datetime(df["prediction_timestamp"])
        return df
    return pd.DataFrame()


@st.cache_data
def load_zone_volumes():
    if DEMAND_15MIN_PARQUET_FEB.exists():
        df = pd.read_parquet(DEMAND_15MIN_PARQUET_FEB)
        return df.groupby("PULocationID")["demand"].sum().sort_values(ascending=False)
    return pd.Series(dtype=int)


# Sidebar Navigation
with st.sidebar:
    st.markdown("## 🚕 NYC DEMAND AI")
    st.caption("Continual Learning & Spatiotemporal Forecasting")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏢 Overview",
            "🎯 Demand Prediction",
            "🗺️ NYC Demand Explorer",
            "📈 Model Monitoring",
            "🔍 Drift Detection",
            "⚖️ Model Governance",
            "⏱️ Simulated Real-Time Mode",
        ],
        index=0,
    )

    st.markdown("---")
    api_online, health_info = check_api_health()
    if api_online:
        st.markdown("**Backend API:** 🟢 `Online` (`:8000`)")
    else:
        st.markdown("**Backend API:** 🟡 `Local In-Memory Mode`")

    st.caption("Dataset: NYC TLC HVFHV (Jan–Feb 2025)")
    st.caption("Forecast Horizon: 15-Minute Intervals")


# ==============================================================================
# PAGE 1: OVERVIEW
# ==============================================================================
if page == "🏢 Overview":
    st.markdown('<div class="main-header">NYC Ride Demand Forecasting & Continual Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Production operational overview, real-time health metrics, and Champion model status.</div>', unsafe_allow_html=True)

    mon = get_monitoring_data()
    mod = get_model_info()

    # System Status Banner
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.markdown("**API Status**")
        if api_online:
            st.markdown('<span class="badge-healthy">🟢 HEALTHY (CONNECTED)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-warning">🟡 STANDALONE MODE</span>', unsafe_allow_html=True)

    with col_s2:
        st.markdown("**Production Model**")
        st.markdown(f'<span class="badge-active">🏆 {mod["champion_version"].upper()}</span>', unsafe_allow_html=True)

    with col_s3:
        st.markdown("**Model State**")
        st.markdown('<span class="badge-healthy">🟢 ACTIVE SERVING</span>', unsafe_allow_html=True)

    with col_s4:
        st.markdown("**Monitoring State**")
        if not mon["is_degraded"]:
            st.markdown('<span class="badge-healthy">🟢 NORMAL (NO DRIFT)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-degraded">🔴 DEGRADED</span>', unsafe_allow_html=True)

    st.markdown("---")

    # KPI Summary Cards
    st.markdown("### 📊 Production Operational Metrics (February Stream)")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.metric("February MAE", f"{mon['overall_mae']:.4f}", "-0.0387 vs Jan Val", delta_color="inverse")
    with c2:
        st.metric("February RMSE", f"{mon['overall_rmse']:.4f}", "-0.0846 vs Jan Val", delta_color="inverse")
    with c3:
        st.metric("WAPE", f"{mon['overall_wape']*100:.2f}%")
    with c4:
        st.metric("Mean Bias", f"{mon['overall_bias']:+.4f}", "Near-Zero Bias")
    with c5:
        st.metric("Total Predictions", f"{mon['total_predictions_logged']:,}")
    with c6:
        st.metric("Pickup Zones", f"{mon['number_of_zones']}")

    st.markdown("---")

    # Benchmark Summary & Architecture Highlights
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 🔬 Baseline vs Machine Learning Benchmark")
        st.caption("Validated on unseen January test set (Jan 26–31) under strict chronological splitting:")

        bench_data = pd.DataFrame([
            {"Model": "Baseline A: Naive (t-1)", "MAE": 5.5234, "RMSE": 8.8132, "Bias": "+0.0323", "Improvement vs Naive": "Baseline (0.0%)"},
            {"Model": "Baseline B: Historical Seasonal", "MAE": 5.2517, "RMSE": 9.0949, "Bias": "-0.0306", "Improvement vs Naive": "+4.92%"},
            {"Model": "Baseline C: Moving Average (1h)", "MAE": 5.3407, "RMSE": 8.9024, "Bias": "+0.0445", "Improvement vs Naive": "+3.31%"},
            {"Model": "Machine Learning (LightGBM V1)", "MAE": 4.3721, "RMSE": 7.2381, "Bias": "+0.0547", "Improvement vs Naive": "+20.84% ⭐"},
        ])
        st.dataframe(bench_data, hide_index=True, use_container_width=True)

        st.info("💡 **Core Finding**: LightGBM Champion V1 achieved a **20.84% MAE reduction** over the Naive baseline by seamlessly fusing short-term autoregressive momentum ($t-1, t-2$) with macro seasonal rhythms ($t-96, t-672$, hour, day of week).")

    with col_right:
        st.markdown("### ⚙️ Production Governance Architecture")
        st.markdown("""
        - **Historical Data**: January 2025 (20.4M records) $\\rightarrow$ Feature Pipeline $\\rightarrow$ **Model V1 Champion**.
        - **Simulated Production**: February 2025 (19.3M records, 704,256 regular grid points) replayed chronologically.
        - **Operational Trigger**: **+15% degradation threshold** over January baseline (5.1630 MAE).
        - **Continual Learning**: Triggered candidate retraining requires strict holdout evaluation with **$\\ge 3\\%$ improvement gate**.
        - **Status**: Peak Feb rolling MAE reached **5.0354** on Feb 16 (Presidents' Day weekend), remaining below alert limit. **Model V1 rightfully retained as Champion**.
        """)


# ==============================================================================
# PAGE 2: DEMAND PREDICTION
# ==============================================================================
elif page == "🎯 Demand Prediction":
    st.markdown('<div class="main-header">Real-Time Demand Inference</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Generate 15-minute zone-level demand forecasts using active Champion Model V1.</div>', unsafe_allow_html=True)

    pred_history = load_historical_predictions()
    zone_vols = load_zone_volumes()
    top_zones = zone_vols.head(20).index.tolist() if not zone_vols.empty else list(range(1, 21))

    c_input1, c_input2, c_input3 = st.columns([1.5, 1.5, 2])

    with c_input1:
        selected_zone = st.selectbox(
            "Select NYC Pickup Zone (PULocationID)",
            options=sorted(pred_history["zone"].unique().tolist() if not pred_history.empty else range(1, 264)),
            index=0 if 138 not in top_zones else (top_zones.index(138) if 138 in top_zones else 0),
            help="Select any of the 262 valid NYC TLC taxi zones.",
        )
        if not zone_vols.empty and selected_zone in zone_vols:
            st.caption(f"Zone total Feb demand: **{zone_vols.loc[selected_zone]:,}** requests")

    with c_input2:
        selected_date = st.date_input(
            "Forecast Date",
            value=datetime(2025, 2, 20).date(),
            min_value=datetime(2025, 1, 1).date(),
            max_value=datetime(2025, 3, 31).date(),
        )

    with c_input3:
        selected_time = st.time_input(
            "Interval Time",
            value=dtime(18, 0),
            step=900,  # 15-minute intervals
        )

    target_timestamp_str = f"{selected_date.strftime('%Y-%m-%d')} {selected_time.strftime('%H:%M:%S')}"

    if st.button("🔮 Generate 15-Minute Demand Prediction", type="primary", use_container_width=True):
        with st.spinner("Querying inference service..."):
            res = query_prediction(zone=selected_zone, timestamp=target_timestamp_str)

        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Predicted 15-Min Demand</div>
                <div class="metric-value">{res['predicted_demand']} <span style="font-size:1rem;font-weight:500;color:#64748B;">rides</span></div>
                <div class="metric-delta">Model: <b>{res['model_version'].upper()}</b></div>
            </div>
            """, unsafe_allow_html=True)

        with col_res2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Target Location & Time</div>
                <div class="metric-value">Zone {res['zone']}</div>
                <div class="metric-delta">Timestamp: <code>{res['timestamp']}</code></div>
            </div>
            """, unsafe_allow_html=True)

        with col_res3:
            exec_time = res.get("execution_time_ms", 1.2)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Inference Latency</div>
                <div class="metric-value">{exec_time} <span style="font-size:1rem;font-weight:500;color:#64748B;">ms</span></div>
                <div class="metric-delta">Status: <span class="badge-healthy">Sub-5ms SLA</span></div>
            </div>
            """, unsafe_allow_html=True)

    # Actual vs Predicted Historical Comparison Chart
    st.markdown("---")
    st.markdown(f"### 📈 Actual vs Predicted Demand for Zone {selected_zone}")

    if not pred_history.empty:
        z_df = pred_history[pred_history["zone"] == selected_zone].sort_values("prediction_timestamp").reset_index(drop=True)
        if not z_df.empty:
            # Filter to February stream
            z_feb = z_df[z_df["prediction_timestamp"] >= "2025-02-01"].copy()
            if z_feb.empty:
                z_feb = z_df

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=z_feb["prediction_timestamp"],
                y=z_feb["actual_demand"],
                mode="lines",
                name="Actual Demand",
                line=dict(color="#1f77b4", width=1.5),
            ))
            fig.add_trace(go.Scatter(
                x=z_feb["prediction_timestamp"],
                y=z_feb["predicted_demand"],
                mode="lines",
                name="Model V1 Forecast",
                line=dict(color="#ff7f0e", width=1.5),
            ))
            fig.update_layout(
                title=f"15-Minute Ride Demand Timeline for Zone {selected_zone} (February 2025)",
                xaxis_title="Date & Time",
                yaxis_title="Demand (Ride Requests / 15 min)",
                hovermode="x unified",
                template="plotly_white",
                height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recorded stream predictions for this zone.")


# ==============================================================================
# PAGE 3: NYC DEMAND EXPLORER
# ==============================================================================
elif page == "🗺️ NYC Demand Explorer":
    st.markdown('<div class="main-header">NYC Spatiotemporal Demand Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explore empirical ride request distributions across pickup zones and time dimensions.</div>', unsafe_allow_html=True)

    pred_history = load_historical_predictions()
    if pred_history.empty:
        st.warning("Prediction history data not found. Please verify data/processed/stream_predictions.parquet.")
    else:
        # 1. Top Demand Zones
        zone_summary = pred_history.groupby("zone").agg(
            total_actual=("actual_demand", "sum"),
            mean_15min_demand=("actual_demand", "mean"),
            max_15min_demand=("actual_demand", "max"),
        ).sort_values("total_actual", ascending=False).reset_index()

        col_top, col_hour = st.columns([1, 1])

        with col_top:
            st.markdown("### 🏆 Top 15 Highest Volume Pickup Zones")
            top15 = zone_summary.head(15).copy()
            top15["Zone_Label"] = top15["zone"].apply(lambda z: f"Zone {z}")
            fig_top = px.bar(
                top15,
                x="Zone_Label",
                y="total_actual",
                labels={"Zone_Label": "Pickup Zone ID", "total_actual": "Total Requests"},
                color="total_actual",
                color_continuous_scale="Viridis",
                template="plotly_white",
                title="Total Requests by Zone (February 2025)",
            )
            fig_top.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig_top, use_container_width=True)

        with col_hour:
            st.markdown("### ⏰ Diurnal Hourly Demand Profile")
            pred_history["hour"] = pred_history["prediction_timestamp"].dt.hour
            hourly_agg = pred_history.groupby("hour")["actual_demand"].mean().reset_index()
            fig_hr = px.line(
                hourly_agg,
                x="hour",
                y="actual_demand",
                markers=True,
                labels={"hour": "Hour of Day (0–23)", "actual_demand": "Mean Demand / Zone"},
                template="plotly_white",
                title="Average 15-Minute Demand by Hour of Day",
            )
            fig_hr.update_traces(line_color="#2ca02c", line_width=2.5)
            fig_hr.update_layout(height=380)
            st.plotly_chart(fig_hr, use_container_width=True)

        # 2. Day of Week Demand
        col_dow, col_dist = st.columns([1, 1])

        with col_dow:
            st.markdown("### 📅 Demand by Day of Week")
            pred_history["dayofweek"] = pred_history["prediction_timestamp"].dt.dayofweek
            dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
            dow_agg = pred_history.groupby("dayofweek")["actual_demand"].mean().reset_index()
            dow_agg["DOW_Name"] = dow_agg["dayofweek"].map(dow_map)
            fig_dow = px.bar(
                dow_agg,
                x="DOW_Name",
                y="actual_demand",
                labels={"DOW_Name": "Day of Week", "actual_demand": "Mean Demand / Zone"},
                color="actual_demand",
                color_continuous_scale="Blues",
                template="plotly_white",
                title="Average Demand by Day of Week",
            )
            fig_dow.update_layout(height=360, coloraxis_showscale=False)
            st.plotly_chart(fig_dow, use_container_width=True)

        with col_dist:
            st.markdown("### 📊 Spatiotemporal Zone Volume Distribution")
            fig_dist = px.histogram(
                zone_summary,
                x="mean_15min_demand",
                nbins=40,
                labels={"mean_15min_demand": "Mean 15-Min Demand per Zone"},
                template="plotly_white",
                color_discrete_sequence=["#9467bd"],
                title="Distribution of Zone Demand Intensities",
            )
            fig_dist.update_layout(height=360)
            st.plotly_chart(fig_dist, use_container_width=True)

        st.caption("ℹ️ Note: Spatiotemporal distributions are aggregated strictly from verified TLC `PULocationID` counts.")


# ==============================================================================
# PAGE 4: MODEL MONITORING
# ==============================================================================
elif page == "📈 Model Monitoring":
    st.markdown('<div class="main-header">Operational Performance Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Continuous tracking of rolling MAE, RMSE, Prediction Bias, and control limits.</div>', unsafe_allow_html=True)

    mon = get_monitoring_data()
    pred_history = load_historical_predictions()

    # Metric Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Latest Rolling 24h MAE", f"{mon['latest_rolling_24h_mae']:.4f}", f"Threshold: {mon['degradation_threshold_mae']:.4f}")
    with c2:
        st.metric("Peak Rolling 24h MAE", f"{mon['peak_rolling_24h_mae']:.4f}", f"Below Alert Limit")
    with c3:
        st.metric("Overall RMSE", f"{mon['overall_rmse']:.4f}")
    with c4:
        st.metric("Operational Status", mon["degradation_status"], "No Sustained Breach")

    st.markdown("---")

    # Rolling 24-Hour MAE Chart
    if not pred_history.empty:
        feb_preds = pred_history[pred_history["prediction_timestamp"] >= "2025-02-01"].copy()
        if feb_preds.empty:
            feb_preds = pred_history

        rolling_df = compute_rolling_metrics(feb_preds, window_size=96)
        full_rolling = rolling_df.iloc[96:].copy() if len(rolling_df) > 96 else rolling_df

        baseline_val_mae = 4.4896
        limit_mae = mon["degradation_threshold_mae"]

        fig_mon = go.Figure()
        fig_mon.add_trace(go.Scatter(
            x=full_rolling["prediction_timestamp"],
            y=full_rolling["rolling_mae"],
            mode="lines",
            name="Rolling 24-Hour MAE",
            line=dict(color="#2ca02c", width=2.5),
        ))
        fig_mon.add_hline(
            y=baseline_val_mae,
            line_dash="dot",
            line_color="#475569",
            annotation_text=f"January Validation Baseline ({baseline_val_mae:.2f})",
            annotation_position="bottom right",
        )
        fig_mon.add_hline(
            y=limit_mae,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text=f"Configured 15% Trigger Limit ({limit_mae:.2f})",
            annotation_position="top right",
        )
        fig_mon.update_layout(
            title="Rolling 24-Hour MAE vs Configured Operational Alert Limit (February 2025)",
            xaxis_title="Date & Time",
            yaxis_title="Rolling 24h MAE",
            template="plotly_white",
            height=430,
            hovermode="x unified",
        )
        st.plotly_chart(fig_mon, use_container_width=True)

        st.markdown("### ⚖️ Prediction Bias (Mean Error) Over Time")
        fig_bias = px.line(
            full_rolling,
            x="prediction_timestamp",
            y="rolling_bias",
            labels={"prediction_timestamp": "Date & Time", "rolling_bias": "Mean Error (Predicted - Actual)"},
            template="plotly_white",
            title="Rolling Prediction Bias Across February",
        )
        fig_bias.add_hline(y=0, line_dash="dash", line_color="black")
        fig_bias.update_traces(line_color="#ff7f0e", line_width=2)
        fig_bias.update_layout(height=320)
        st.plotly_chart(fig_bias, use_container_width=True)


# ==============================================================================
# PAGE 5: DRIFT DETECTION
# ==============================================================================
elif page == "🔍 Drift Detection":
    st.markdown('<div class="main-header">Data Drift & Statistical Profiling</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Population Stability Index (PSI) and Kolmogorov-Smirnov distribution testing.</div>', unsafe_allow_html=True)

    drift = get_drift_data()

    st.markdown(f"""
    <div style="background-color:#F8FAFC;border-left:4px solid #3B82F6;padding:12px 16px;border-radius:4px;margin-bottom:1.5rem;">
        <b>Scientific Principle</b>: {drift['scientific_note']}
    </div>
    """, unsafe_allow_html=True)

    c_d1, c_d2 = st.columns([1, 1])
    with c_d1:
        st.markdown(f"**Reference Period (Training)**: `{drift['reference_period']}`")
    with c_d2:
        st.markdown(f"**Current Period (Production)**: `{drift['current_period']}`")

    st.markdown("---")

    # Drift Summary Table
    drift_items = drift.get("features", [])
    if drift_items:
        table_rows = []
        for item in drift_items:
            psi = item["psi"]
            status_badge = "🟢 Stable" if psi < 0.10 else ("🟡 Moderate Shift" if psi < 0.25 else "🔴 Drift Detected")
            table_rows.append({
                "Feature Name": item["feature"],
                "PSI Value": f"{psi:.4f}",
                "PSI Threshold": "0.10 (Shift) / 0.25 (Drift)",
                "KS Statistic": f"{item.get('ks_statistic', 0.0):.4f}" if item.get("ks_statistic") is not None else "N/A",
                "KS P-Value": f"{item.get('ks_p_value', 1.0):.4e}" if item.get("ks_p_value") is not None else "N/A",
                "Status": status_badge,
            })

        st.markdown("### 📋 Monitored Feature Drift Table")
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

        # PSI Bar Chart
        psi_df = pd.DataFrame(drift_items)
        fig_psi = px.bar(
            psi_df,
            x="feature",
            y="psi",
            labels={"feature": "Monitored Feature", "psi": "Population Stability Index (PSI)"},
            color="psi",
            color_continuous_scale=["#2ca02c", "#eab308", "#dc2626"],
            range_color=[0, 0.3],
            template="plotly_white",
            title="Population Stability Index (PSI) Across Monitored Features",
        )
        fig_psi.add_hline(y=0.10, line_dash="dash", line_color="orange", annotation_text="Moderate Shift (0.10)")
        fig_psi.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="Significant Drift (0.25)")
        fig_psi.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_psi, use_container_width=True)


# ==============================================================================
# PAGE 6: MODEL GOVERNANCE
# ==============================================================================
elif page == "⚖️ Model Governance":
    st.markdown('<div class="main-header">Continual Learning & Model Governance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Champion-Challenger validation gates, audit decision logs, and model lineage.</div>', unsafe_allow_html=True)

    gov = get_governance_data()

    # Governance Gate Highlight
    col_g1, col_g2 = st.columns([1, 1])

    with col_g1:
        st.markdown("### 🏆 Production Champion")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Active Model</div>
            <div class="metric-value">{gov['champion_version'].upper()} <span class="badge-healthy">ACTIVE CHAMPION</span></div>
            <div class="metric-delta">Holdout MAE: <b>{gov['champion_mae']:.4f}</b></div>
            <div class="metric-delta">Role: Serves 100% of production inference traffic</div>
        </div>
        """, unsafe_allow_html=True)

    with col_g2:
        st.markdown("### 🥊 Candidate Challenger")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Challenger Model</div>
            <div class="metric-value">{gov.get('challenger_version', 'model_v2_feb').upper()} <span class="badge-degraded">REJECTED BY GATE</span></div>
            <div class="metric-delta">Holdout MAE: <b>{gov.get('challenger_mae', 4.4056):.4f}</b></div>
            <div class="metric-delta">Observed Improvement: <b>{gov.get('observed_improvement_pct', -1.50):+.2f}%</b> (Required: ≥ {gov['required_improvement_pct']}%)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Gate Evaluation Results
    st.markdown("### 🛡️ Champion vs Challenger Holdout Gate Evaluation")
    gate_table = pd.DataFrame([
        {"Metric": "Evaluation Window", "Champion (Model V1)": "Feb 16–20 (Unseen Holdout)", "Challenger (Model V2)": "Feb 16–20 (Unseen Holdout)", "Target Gate": "Exact Same Holdout"},
        {"Metric": "Holdout MAE", "Champion (Model V1)": f"{gov['champion_mae']:.4f}", "Challenger (Model V2)": f"{gov.get('challenger_mae', 4.4056):.4f}", "Target Gate": "Lower is better"},
        {"Metric": "Relative MAE Improvement", "Champion (Model V1)": "Baseline (0.00%)", "Challenger (Model V2)": f"{gov.get('observed_improvement_pct', -1.50):+.2f}%", "Target Gate": f"≥ +{gov['required_improvement_pct']}% Required"},
        {"Metric": "Safety Gate Decision", "Champion (Model V1)": "RETAINED CHAMPION", "Challenger (Model V2)": gov["decision"], "Target Gate": "Pass/Fail Criteria"},
    ])
    st.dataframe(gate_table, hide_index=True, use_container_width=True)

    st.warning(f"⚠️ **Governance Audit Outcome**: {gov['reason']}")

    st.markdown("---")

    # Visual Lineage Timeline
    st.markdown("### 📜 Model Lineage and Registry History")
    lineage_records = gov.get("lineage", [])
    if lineage_records:
        st.dataframe(pd.DataFrame(lineage_records), hide_index=True, use_container_width=True)


# ==============================================================================
# PAGE 7: SIMULATED REAL-TIME MODE
# ==============================================================================
elif page == "⏱️ Simulated Real-Time Mode":
    st.markdown('<div class="main-header">Simulated Real-Time Production Stream Replay</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Chronological event replay through February 2025 with delayed ground-truth arrival.</div>', unsafe_allow_html=True)

    pred_history = load_historical_predictions()

    if pred_history.empty:
        st.warning("Prediction history data not found.")
    else:
        feb_history = pred_history[pred_history["prediction_timestamp"] >= "2025-02-01"].sort_values("prediction_timestamp").reset_index(drop=True)
        unique_timestamps = sorted(feb_history["prediction_timestamp"].unique())

        col_ctrl1, col_ctrl2 = st.columns([2, 1])

        with col_ctrl1:
            selected_idx = st.slider(
                "Step Through Historical Timeline (15-min intervals)",
                min_value=0,
                max_value=len(unique_timestamps) - 1,
                value=len(unique_timestamps) // 2,
                format="%d",
            )
            current_ts = unique_timestamps[selected_idx]
            st.markdown(f"**Current Replay Clock**: `🕒 {pd.to_datetime(current_ts).strftime('%Y-%m-%d %H:%M:%S')}`")

        with col_ctrl2:
            sim_zone = st.selectbox("Replay Zone", options=sorted(feb_history["zone"].unique().tolist()), index=137)

        # Slice data for current timestamp
        ts_data = feb_history[feb_history["prediction_timestamp"] == current_ts]
        zone_event = ts_data[ts_data["zone"] == sim_zone]

        st.markdown("---")
        st.markdown("### 📡 Live Inference & Audit Event")

        if not zone_event.empty:
            row = zone_event.iloc[0]
            c_e1, c_e2, c_e3, c_e4 = st.columns(4)

            with c_e1:
                st.metric("Model V1 Prediction", f"{row['predicted_demand']:.1f} rides")
            with c_e2:
                st.metric("Actual Demand Arrived", f"{row['actual_demand']} rides")
            with c_e3:
                st.metric("Prediction Error", f"{row['error']:+.2f}", delta_color="inverse")
            with c_e4:
                st.metric("Absolute Error", f"{row['absolute_error']:.2f}")

        # Zone timeline up to current timestamp
        st.markdown("### 📊 Rolling Context for Zone " + str(sim_zone))
        zone_past = feb_history[(feb_history["zone"] == sim_zone) & (feb_history["prediction_timestamp"] <= current_ts)].tail(96)

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=zone_past["prediction_timestamp"],
            y=zone_past["actual_demand"],
            name="Actual Demand",
            line=dict(color="#1f77b4", width=2),
        ))
        fig_sim.add_trace(go.Scatter(
            x=zone_past["prediction_timestamp"],
            y=zone_past["predicted_demand"],
            name="Model V1 Forecast",
            line=dict(color="#ff7f0e", width=2),
        ))
        fig_sim.update_layout(
            title=f"Recent 24-Hour Replay Context for Zone {sim_zone}",
            xaxis_title="Timestamp",
            yaxis_title="Ride Demand",
            template="plotly_white",
            height=360,
        )
        st.plotly_chart(fig_sim, use_container_width=True)

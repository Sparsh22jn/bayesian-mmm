"""Tab 2 — Response curves: saturation curves for controllable channels only."""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.mmm.optimize import CONTROLLABLE_COLS, FIXED_COLS


def render():
    st.header("Response Curves")
    st.caption(
        "Diminishing-returns (Hill saturation) curves for **controllable** channels. "
        "These show how much incremental Sales each additional impression/view delivers."
    )

    st.info(
        f"Curves are shown only for controllable channels: "
        f"{', '.join(f'`{c}`' for c in CONTROLLABLE_COLS)}.  \n"
        f"Fixed channels (`{'`, `'.join(FIXED_COLS)}`) are excluded — "
        "optimising their volumes is not actionable.",
        icon="ℹ️",
    )

    if "mmm" not in st.session_state or st.session_state.get("idata") is None:
        st.info("Fit the model first to see posterior response curves.")
        _demo_curve()
        return

    optimizer = st.session_state.get("optimizer")
    if optimizer is None:
        st.warning("Optimiser not initialised. Re-fit the model.")
        return

    df = st.session_state.get("df")
    channel = st.selectbox("Select channel", optimizer.controllable_cols)

    default_max = int(df[channel].mean() * 3) if df is not None else 2_000_000
    max_vol = st.number_input(
        "Max volume on x-axis",
        value=default_max,
        step=10_000,
        help="Upper bound for the x-axis (impressions / views).",
    )

    volume_range = np.linspace(0, max_vol, 300)
    try:
        curve_df = optimizer.roas_curve(channel, volume_range)
    except ValueError as e:
        st.error(str(e))
        return

    current_vol = df[channel].mean() if df is not None else max_vol / 2

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve_df["volume"],
        y=curve_df["response"],
        mode="lines",
        name="Mean posterior response",
        line=dict(width=2.5),
    ))
    fig.add_vline(
        x=current_vol,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean volume ({current_vol:,.0f})",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="Impressions / Views",
        yaxis_title="Normalised Response (0–1)",
        title=f"Saturation Curve — {channel}",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Marginal response at current volume
    idx = np.searchsorted(curve_df["volume"], current_vol)
    if 0 < idx < len(curve_df) - 1:
        marginal = (curve_df["response"].iloc[idx + 1] - curve_df["response"].iloc[idx - 1]) / (
            curve_df["volume"].iloc[idx + 1] - curve_df["volume"].iloc[idx - 1]
        )
        saturation_pct = curve_df["response"].iloc[idx] * 100
        col1, col2 = st.columns(2)
        col1.metric("Saturation at current volume", f"{saturation_pct:.1f}%")
        col2.metric("Marginal response per 1k impressions", f"{marginal * 1000:.4f}")


def _demo_curve():
    vol = np.linspace(0, 2_000_000, 300)
    fig = go.Figure()
    for label, k in [
        ("Paid_Views (k=30k)", 30_000),
        ("Google_Impressions (k=900k)", 900_000),
        ("Email_Impressions (k=700k)", 700_000),
        ("Facebook_Impressions (k=250k)", 250_000),
    ]:
        fig.add_trace(go.Scatter(x=vol, y=vol / (k + vol), mode="lines", name=label))
    fig.update_layout(
        xaxis_title="Volume (impressions / views)",
        yaxis_title="Normalised Response",
        title="Demo — Hill Saturation Curves (controllable channels only)",
    )
    st.plotly_chart(fig, use_container_width=True)

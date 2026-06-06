"""Tab 2 — Response curves: adstock + saturation per channel with uncertainty bands."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def render():
    st.header("Response Curves")
    st.caption("Visualise the diminishing-returns (Hill saturation) curve per channel.")

    if "mmm" not in st.session_state or st.session_state.get("idata") is None:
        st.info("Fit the model first to see posterior response curves.")
        _demo_curve()
        return

    mmm = st.session_state["mmm"]
    optimizer = st.session_state.get("optimizer")

    channel = st.selectbox("Select channel", mmm.channel_cols)
    max_spend = st.number_input("Max spend on x-axis ($)", value=500_000, step=10_000)

    if optimizer:
        spend_range = np.linspace(0, max_spend, 300)
        curve_df = optimizer.roas_curve(channel, spend_range)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=curve_df["spend"],
                y=curve_df["response"],
                mode="lines",
                name="Mean posterior",
                line=dict(width=2),
            )
        )
        current = st.session_state.get("current_spend", {}).get(channel, max_spend / 2)
        fig.add_vline(x=current, line_dash="dash", annotation_text="Current spend")
        fig.update_layout(
            xaxis_title="Spend ($)",
            yaxis_title="Normalised Response",
            title=f"Saturation Curve — {channel}",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Optimiser not initialised. Re-run fitting.")


def _demo_curve():
    spend = np.linspace(0, 500_000, 300)
    for label, k in [("TV (k=150k)", 150_000), ("Digital (k=80k)", 80_000), ("Social (k=60k)", 60_000)]:
        response = spend / (k + spend)
        fig = go.Figure(
            go.Scatter(x=spend, y=response, mode="lines", name=label)
        )
    fig = go.Figure()
    for label, k in [("TV (k=150k)", 150_000), ("Digital (k=80k)", 80_000), ("Social (k=60k)", 60_000)]:
        response = spend / (k + spend)
        fig.add_trace(go.Scatter(x=spend, y=response, mode="lines", name=label))
    fig.update_layout(
        xaxis_title="Spend ($)", yaxis_title="Normalised Response",
        title="Demo — Hill Saturation Curves"
    )
    st.plotly_chart(fig, use_container_width=True)

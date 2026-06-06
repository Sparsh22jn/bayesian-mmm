"""Tab 1 — Model overview: channel contribution breakdown and KPI decomposition."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render():
    st.header("Model Overview")

    if "idata" not in st.session_state or st.session_state["idata"] is None:
        st.info("Upload your data and run the model from the sidebar to see results.")
        _show_demo()
        return

    idata = st.session_state["idata"]
    mmm = st.session_state["mmm"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Channels modelled", len(mmm.channel_cols))
    with col2:
        st.metric("Weeks of data", len(st.session_state.get("df", [])))
    with col3:
        r2 = st.session_state.get("r2", "—")
        st.metric("In-sample R²", f"{r2:.3f}" if isinstance(r2, float) else r2)

    st.subheader("Channel Contribution Share")
    try:
        contrib = mmm.channel_contribution_breakdown()
        fig = px.bar(
            contrib,
            x="channel",
            y="contribution",
            color="channel",
            labels={"contribution": "Mean Posterior Contribution"},
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render contributions: {e}")

    st.subheader("Actual vs Fitted")
    if "df" in st.session_state and "fitted" in st.session_state:
        df = st.session_state["df"]
        fitted = st.session_state["fitted"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["date"], y=df[mmm.target_col], name="Actual"))
        fig2.add_trace(go.Scatter(x=df["date"], y=fitted, name="Fitted", line=dict(dash="dash")))
        st.plotly_chart(fig2, use_container_width=True)


def _show_demo():
    st.subheader("Demo — Simulated Channel Contributions")
    demo = pd.DataFrame(
        {
            "channel": ["TV", "Digital", "Social", "OOH", "Base"],
            "contribution": [0.28, 0.35, 0.18, 0.09, 0.10],
        }
    )
    fig = px.pie(demo, values="contribution", names="channel", title="Contribution Share (demo)")
    st.plotly_chart(fig, use_container_width=True)

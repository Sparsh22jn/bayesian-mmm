"""Tab 3 — Budget optimizer: interactive spend allocation under a total budget constraint."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px


def render():
    st.header("Budget Optimizer")
    st.caption("Maximise predicted KPI by reallocating spend across channels.")

    if "mmm" not in st.session_state or st.session_state.get("idata") is None:
        st.info("Fit the model first to unlock budget optimisation.")
        _demo_table()
        return

    mmm = st.session_state["mmm"]
    optimizer = st.session_state.get("optimizer")

    total_budget = st.number_input(
        "Total weekly budget ($)", min_value=1_000, value=200_000, step=5_000
    )

    st.subheader("Per-channel bounds")
    lower, upper, current = [], [], []
    cols = st.columns(3)
    for i, ch in enumerate(mmm.channel_cols):
        with cols[0]:
            lower.append(st.number_input(f"{ch} min", value=0, key=f"lb_{ch}"))
        with cols[1]:
            current.append(
                st.number_input(f"{ch} current", value=int(total_budget / len(mmm.channel_cols)), key=f"cur_{ch}")
            )
        with cols[2]:
            upper.append(st.number_input(f"{ch} max", value=total_budget, key=f"ub_{ch}"))

    if st.button("Run optimisation", type="primary"):
        if optimizer is None:
            st.error("Optimiser not available. Re-fit the model.")
            return
        result = optimizer.optimize(
            total_budget=total_budget,
            lower_bounds=lower,
            upper_bounds=upper,
            current_spend=np.array(current, dtype=float),
        )
        st.session_state["opt_result"] = result

    if "opt_result" in st.session_state:
        result = st.session_state["opt_result"]
        st.subheader("Optimal Allocation")
        st.dataframe(result.style.format({"optimal_spend": "${:,.0f}", "current_spend": "${:,.0f}", "delta": "${:+,.0f}"}))

        fig = px.bar(
            result.melt(id_vars="channel", value_vars=["current_spend", "optimal_spend"]),
            x="channel", y="value", color="variable", barmode="group",
            labels={"value": "Spend ($)"},
            title="Current vs Optimal Spend",
        )
        st.plotly_chart(fig, use_container_width=True)


def _demo_table():
    demo = pd.DataFrame({
        "channel": ["TV", "Digital", "Social", "OOH"],
        "current_spend": [60_000, 80_000, 40_000, 20_000],
        "optimal_spend": [45_000, 100_000, 45_000, 10_000],
        "delta": [-15_000, 20_000, 5_000, -10_000],
    })
    st.subheader("Demo — Optimised Allocation")
    st.dataframe(demo)

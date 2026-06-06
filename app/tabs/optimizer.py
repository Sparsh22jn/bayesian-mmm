"""Tab 3 — Budget optimizer: allocate impressions across controllable channels only."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from src.mmm.optimize import CONTROLLABLE_COLS, FIXED_COLS


def render():
    st.header("Budget Optimizer")
    st.caption(
        "Reallocate impression/view volume across **controllable** channels to maximise predicted Sales."
    )

    with st.expander("Which channels are included and why?"):
        st.markdown(
            f"**Optimised (controllable):** {', '.join(f'`{c}`' for c in CONTROLLABLE_COLS)}\n\n"
            f"**Held fixed:** {', '.join(f'`{c}`' for c in FIXED_COLS)}  \n"
            "`Organic_Views` is driven by content quality, not spend. "
            "`Affiliate_Impressions` are performance-based — partners control their own traffic. "
            "Both are included in the MMM so their contribution is measured, but they are not dials we can turn."
        )

    if "mmm" not in st.session_state or st.session_state.get("idata") is None:
        st.info("Fit the model first to unlock budget optimisation.")
        _demo_table()
        return

    optimizer = st.session_state.get("optimizer")
    if optimizer is None:
        st.error("Optimiser not initialised. Re-fit the model.")
        return

    controllable = optimizer.controllable_cols
    n = len(controllable)

    # Use mean historical volumes as defaults
    df = st.session_state.get("df")
    if df is not None:
        default_volumes = [int(df[ch].mean()) for ch in controllable]
    else:
        default_volumes = [100_000] * n

    total_default = sum(default_volumes)
    total_budget = st.number_input(
        "Total weekly impression/view budget",
        min_value=1_000,
        value=total_default,
        step=10_000,
        help="Total volume to distribute across the 4 controllable channels.",
    )

    st.subheader("Per-channel bounds")
    st.caption("Set the min and max volume each channel can receive.")
    lower, upper, current = [], [], []
    header_cols = st.columns([2, 1, 1, 1])
    header_cols[0].markdown("**Channel**")
    header_cols[1].markdown("**Min**")
    header_cols[2].markdown("**Current (baseline)**")
    header_cols[3].markdown("**Max**")

    for i, ch in enumerate(controllable):
        row = st.columns([2, 1, 1, 1])
        row[0].markdown(f"`{ch}`")
        lower.append(row[1].number_input("", value=0, key=f"lb_{ch}", label_visibility="collapsed"))
        current.append(row[2].number_input("", value=default_volumes[i], key=f"cur_{ch}", label_visibility="collapsed"))
        upper.append(row[3].number_input("", value=total_budget, key=f"ub_{ch}", label_visibility="collapsed"))

    if st.button("Run optimisation", type="primary"):
        result = optimizer.optimize(
            total_budget=float(total_budget),
            lower_bounds=[float(x) for x in lower],
            upper_bounds=[float(x) for x in upper],
            current_volumes=np.array(current, dtype=float),
        )
        st.session_state["opt_result"] = result

    if "opt_result" in st.session_state:
        result = st.session_state["opt_result"]
        st.subheader("Optimal Allocation")

        fmt = {
            "current_volume": "{:,.0f}",
            "optimal_volume": "{:,.0f}",
            "delta": "{:+,.0f}",
            "pct_change": "{:+.1f}%",
        }
        st.dataframe(result.style.format(fmt), use_container_width=True)

        fig = px.bar(
            result.melt(id_vars="channel", value_vars=["current_volume", "optimal_volume"]),
            x="channel",
            y="value",
            color="variable",
            barmode="group",
            color_discrete_map={"current_volume": "#636EFA", "optimal_volume": "#00CC96"},
            labels={"value": "Impressions / Views", "variable": ""},
            title="Current vs Optimal Volume per Controllable Channel",
        )
        st.plotly_chart(fig, use_container_width=True)


def _demo_table():
    demo = pd.DataFrame({
        "channel": CONTROLLABLE_COLS,
        "current_volume": [15_094, 886_174, 760_509, 269_127],
        "optimal_volume":  [10_000, 1_100_000, 600_000, 420_000],
        "delta":           [-5_094, 213_826, -160_509, 150_873],
        "pct_change":      [-33.8, 24.1, -21.1, 56.1],
    })
    st.subheader("Demo — Optimised Allocation (illustrative)")
    st.dataframe(demo, use_container_width=True)

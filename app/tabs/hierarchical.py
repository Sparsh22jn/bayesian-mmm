"""Tab 4 — Hierarchical view: region/brand-level coefficient distributions."""

import streamlit as st
import numpy as np
import pandas as pd
import arviz as az
import plotly.graph_objects as go
import plotly.express as px


def render():
    st.header("Hierarchical Model View")
    st.caption("Posterior distributions of channel coefficients — uncertainty quantified.")

    if "idata" not in st.session_state or st.session_state["idata"] is None:
        st.info("Fit the model first to explore the posterior.")
        _demo_posteriors()
        return

    idata = st.session_state["idata"]
    mmm = st.session_state["mmm"]

    st.subheader("Posterior Coefficient Distributions")
    param_options = [f"saturation_{ch}_lam" for ch in mmm.channel_cols] + \
                    [f"adstock_{ch}_alpha" for ch in mmm.channel_cols]
    selected = st.multiselect("Parameters to display", param_options, default=param_options[:4])

    for param in selected:
        if param in idata.posterior:
            samples = idata.posterior[param].values.flatten()
            fig = go.Figure(go.Histogram(x=samples, nbinsx=60, name=param))
            fig.update_layout(title=param, xaxis_title="Value", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Parameter `{param}` not found in posterior.")

    st.subheader("MCMC Diagnostics")
    if st.button("Show R-hat summary"):
        summary = az.summary(idata, var_names=selected[:8])
        st.dataframe(summary[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat", "ess_bulk"]])


def _demo_posteriors():
    st.subheader("Demo — Simulated Posterior Distributions")
    rng = np.random.default_rng(0)
    channels = ["TV", "Digital", "Social", "OOH"]
    for ch in channels:
        samples = rng.beta(3, 5, 2000)
        fig = go.Figure(go.Histogram(x=samples, nbinsx=50))
        fig.update_layout(title=f"adstock_alpha — {ch} (demo)", xaxis_title="alpha", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

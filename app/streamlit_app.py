"""Main Streamlit entry point — run with: streamlit run app/streamlit_app.py"""

import streamlit as st
import sys
from pathlib import Path

# Make src importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.tabs import overview, response_curves, optimizer, hierarchical, insights

st.set_page_config(
    page_title="Bayesian MMM Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Bayesian Marketing Mix Model")
st.caption("Powered by PyMC-Marketing · Visualised with Streamlit")

tab_labels = [
    "Overview",
    "Response Curves",
    "Budget Optimizer",
    "Hierarchical View",
    "AI Insights",
]
tabs = st.tabs(tab_labels)

with tabs[0]:
    overview.render()

with tabs[1]:
    response_curves.render()

with tabs[2]:
    optimizer.render()

with tabs[3]:
    hierarchical.render()

with tabs[4]:
    insights.render()

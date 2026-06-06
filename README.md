# Bayesian MMM Portfolio Dashboard

A production-ready **Marketing Mix Model** built with [PyMC-Marketing](https://www.pymc-marketing.io/) and visualised in Streamlit.

## Stack

| Layer | Library |
|-------|---------|
| Bayesian inference | PyMC 5 · PyMC-Marketing |
| Posterior analysis | ArviZ |
| Optimisation | SciPy SLSQP |
| Dashboard | Streamlit |
| AI insights | Claude API (Anthropic) |

## Project layout

```
data/raw/          ← drop your Kaggle CSV here (gitignored)
data/processed/    ← cleaned parquet + MCMC trace (.nc)
notebooks/         ← EDA → modelling → optimisation
src/mmm/           ← reusable Python package
app/               ← Streamlit dashboard
```

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Drop your CSV into data/raw/

# 4. Run the notebooks in order
jupyter lab notebooks/

# 5. Launch the dashboard
streamlit run app/streamlit_app.py
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_eda.ipynb` | Explore distributions, correlations, seasonality |
| `02_model.ipynb` | Fit hierarchical Bayesian MMM; inspect posteriors |
| `03_optimize.ipynb` | Plot ROAS curves; run budget optimiser |

## Dashboard tabs

- **Overview** — channel contribution breakdown, actual vs fitted KPI
- **Response Curves** — Hill saturation curves with uncertainty
- **Budget Optimizer** — interactive reallocation under spend constraints
- **Hierarchical View** — posterior distributions, R-hat diagnostics
- **AI Insights** — Claude-powered Q&A over your model results

## Configuration

Add a `.streamlit/secrets.toml` (gitignored) to avoid typing the API key each time:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

Then read it in `insights.py` via `st.secrets["ANTHROPIC_API_KEY"]`.

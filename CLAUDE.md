# Project Guide — Bayesian MMM Portfolio

## What this project is

A hierarchical Bayesian Marketing Mix Model over 26 divisions of weekly media
spend data (2018–2020). The goal is to estimate channel contributions, posterior
uncertainty, and an optimal budget allocation, then surface results in a
Streamlit dashboard.

---

## The non-negotiable rule

**All analysis, modelling, and computation must be written as executable Python
code — in a notebook cell, in a src/ module, or in a test.** Never describe
what code *would* do. Never summarise results by reasoning from training
knowledge. Write the code, let the kernel run it, read the output.

If you are asked to "analyse the data", write cells. If you are asked to
"check model diagnostics", write cells. If the user asks a question that
requires looking at numbers, write a cell that prints those numbers.

---

## Data schema (do not guess column names — use these)

File: `data/raw/Sample Media Spend Data.csv`

| Column | Type | Notes |
|--------|------|-------|
| `Division` | str | A–Z, 26 divisions — the hierarchical grouping |
| `Calendar_Week` | str → datetime | Format `M/D/YYYY`, weekly frequency |
| `Paid_Views` | int | Paid media views |
| `Organic_Views` | int | Organic views — treat as a control, not a spend channel |
| `Google_Impressions` | int | Google ad impressions |
| `Email_Impressions` | float | Email campaign impressions |
| `Facebook_Impressions` | int | Facebook ad impressions |
| `Affiliate_Impressions` | int | Affiliate channel impressions |
| `Overall_Views` | int | Paid + Organic combined — usually omit (collinear) |
| `Sales` | int | **Target variable** |

**Media channels (modelled with adstock + saturation):**
- `Paid_Views`, `Google_Impressions`, `Email_Impressions`, `Facebook_Impressions`, `Affiliate_Impressions`

**Controls (linear, no adstock):** `Organic_Views`, any seasonal features you engineer

**Never use** `Overall_Views` as a feature — it is a sum of Paid + Organic and will cause collinearity.

### Controllable vs fixed — critical for the optimizer

| Channel | Optimisable? | Reason |
|---------|-------------|--------|
| `Paid_Views` | **Yes** | YouTube paid ads — directly purchasable |
| `Google_Impressions` | **Yes** | Google Ads budget — directly purchasable |
| `Email_Impressions` | **Yes** | Emails deployed — we control send volume |
| `Facebook_Impressions` | **Yes** | Facebook Ads budget — directly purchasable |
| `Organic_Views` | **No** | YouTube organic — driven by content, not spend |
| `Affiliate_Impressions` | **No** | Performance-based — partners control their own traffic |

`CONTROLLABLE_COLS` and `FIXED_COLS` are the canonical lists defined in
`src/mmm/optimize.py`. Import from there — do not redefine them elsewhere.

---

## Notebook pipeline — the only valid execution order

Each notebook reads its inputs, produces outputs, and stops. The next notebook
picks up from those outputs. Never skip steps.

```
01_eda.ipynb
  reads : data/raw/Sample Media Spend Data.csv
  writes: data/processed/clean.parquet
          data/processed/eda_stats.json   ← summary stats for the app

02_model.ipynb
  reads : data/processed/clean.parquet
  writes: data/processed/trace.nc         ← ArviZ InferenceData (MCMC trace)
          data/processed/contributions.parquet
          data/processed/model_summary.json

03_optimize.ipynb
  reads : data/processed/trace.nc
          data/processed/clean.parquet
  writes: data/processed/roas_curves.parquet
          data/processed/optimal_allocation.parquet
```

The Streamlit app reads **only** from `data/processed/`. It never imports PyMC,
never samples, never trains. It is a dashboard, not a compute engine.

---

## How to work in each notebook

### 01 — EDA

Goal: understand the data before touching any model.

Mandatory checks before moving to 02:
- [ ] Parse `Calendar_Week` to `datetime` and sort by `(Division, date)`
- [ ] Confirm no missing values in channel or target columns
- [ ] Plot Sales time series per division (small multiples)
- [ ] Plot channel spend distributions (histograms + boxplots)
- [ ] Correlation matrix: channels vs Sales, per division
- [ ] Check for zero-spend weeks per channel (important for adstock)
- [ ] Check stationarity visually (rolling mean on Sales)
- [ ] Save `clean.parquet` with `date` column properly typed

### 02 — Model

Goal: fit a hierarchical Bayesian MMM with partial pooling across divisions.

Use **PyMC-Marketing** `MMM` class with:
- `GeometricAdstock(l_max=8)` — 8-week carry-over window
- `LogisticSaturation()` — S-curve saturation per channel
- Hierarchical priors: divisions share a population-level prior on adstock
  alpha and saturation lambda → partial pooling, not separate models

Sampling settings (start here, tighten if R-hat > 1.05):
```python
draws=1000, tune=1000, target_accept=0.9, random_seed=42
```

Mandatory diagnostics before saving the trace:
- [ ] `az.summary()` — flag any R-hat > 1.05
- [ ] `az.plot_energy()` — check for energy fraction of missing information
- [ ] Posterior predictive check: `mmm.sample_posterior_predictive()`
- [ ] Actual vs fitted plot
- [ ] Channel contribution waterfall chart

### 03 — Optimise

Goal: ROAS curves + budget optimiser, no new sampling.

- Load trace from `trace.nc` — never re-sample in this notebook
- Generate ROAS curves by sweeping spend 0 → 2× current per channel
- Run SLSQP optimiser with total budget = sum of current mean spend
- Save outputs as parquet so the app can load them instantly

---

## Plotting conventions

Always use **matplotlib + seaborn** for notebook EDA (static, reproducible).
Use **plotly** only for the Streamlit app (interactive).

```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120
```

- Small multiples for per-division views: `FacetGrid` or manual `subplots`
- Always label axes and add a title
- Always call `plt.tight_layout()` before showing
- Never use `plt.show()` in notebook cells — let Jupyter render inline
- Save key figures: `fig.savefig('data/processed/fig_name.png', bbox_inches='tight')`

---

## Modelling conventions

- Random seed: always `random_seed=42` for reproducibility
- Scale continuous inputs to [0,1] before modelling (use `src/mmm/transforms.normalize`)
- Never standardise the target — keep Sales in original units for interpretability
- Partial pooling > separate models. With 26 divisions, hierarchical priors are
  the whole point. Do not fit 26 independent models.
- Always inspect `az.summary(idata)` for R-hat and ESS before trusting posteriors
- Save traces as `.nc` (NetCDF) — this is the ArviZ standard and handles
  posterior, prior, and posterior-predictive in one file

---

## What belongs where

| Work | Location |
|------|----------|
| Data loading, cleaning | `01_eda.ipynb` |
| Feature engineering | `01_eda.ipynb` (simple) or `src/mmm/transforms.py` (reused) |
| MCMC sampling | `02_model.ipynb` only |
| Diagnostics | `02_model.ipynb` |
| Budget optimisation | `03_optimize.ipynb` |
| Reusable functions | `src/mmm/` modules |
| Dashboard rendering | `app/` only |
| Raw data | `data/raw/` — never modify, never commit non-.gitkeep files |
| Artifacts for the app | `data/processed/` — parquet, json, nc files |

---

## Import template for every notebook

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path('..').resolve()))  # makes src/ importable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120

RAW   = Path('../data/raw')
PROC  = Path('../data/processed')
PROC.mkdir(exist_ok=True)
```

---

## Claude Code behaviour in this project

- When asked to work in a notebook: write cells, not prose descriptions
- When asked to "check" something: write code that prints or plots the answer
- When asked to "analyse": produce a sequence of cells with one clear question per cell
- When importing from `src/mmm`: check the actual file before writing the import
- When a cell fails: read the traceback, fix the cell, do not suggest running in a different environment
- Do not install packages mid-session — all deps are in `requirements.txt`
- Do not suggest cloud notebooks, Colab, or any external compute
- Do not write cells that call external APIs mid-analysis (the Claude API is only for the insights tab in the app)

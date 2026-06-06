"""Hierarchical Bayesian MMM built on PyMC-Marketing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation


class BayesianMMM:
    """
    Thin wrapper around PyMC-Marketing's MMM class.

    Parameters
    ----------
    channel_cols : list[str]
        Column names for paid media spend variables.
    control_cols : list[str] | None
        Column names for organic/control variables (seasonality, etc.)
    date_col : str
        Column containing weekly/daily dates.
    target_col : str
        Column containing the KPI (revenue, conversions, …).
    """

    def __init__(
        self,
        channel_cols: list[str],
        control_cols: list[str] | None = None,
        date_col: str = "date",
        target_col: str = "revenue",
    ):
        self.channel_cols = channel_cols
        self.control_cols = control_cols or []
        self.date_col = date_col
        self.target_col = target_col
        self.mmm: MMM | None = None
        self.idata: az.InferenceData | None = None

    def build(self, df: pd.DataFrame) -> MMM:
        self.mmm = MMM(
            date_column=self.date_col,
            channel_columns=self.channel_cols,
            control_columns=self.control_cols if self.control_cols else None,
            adstock=GeometricAdstock(l_max=8),
            saturation=LogisticSaturation(),
        )
        return self.mmm

    def fit(
        self,
        df: pd.DataFrame,
        draws: int = 1000,
        tune: int = 1000,
        target_accept: float = 0.9,
        random_seed: int = 42,
    ) -> az.InferenceData:
        if self.mmm is None:
            self.build(df)

        X = df[[self.date_col] + self.channel_cols + self.control_cols]
        y = df[self.target_col].values

        self.idata = self.mmm.fit(
            X=X,
            y=y,
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            random_seed=random_seed,
        )
        return self.idata

    def channel_contribution_breakdown(self) -> pd.DataFrame:
        """Return mean posterior contribution share per channel."""
        if self.idata is None:
            raise RuntimeError("Model not fitted yet.")
        contributions = self.mmm.compute_channel_contribution_original_scale(
            original_scale=True
        )
        mean_contrib = contributions.mean(dim=["chain", "draw"]).to_dataframe().reset_index()
        return mean_contrib

    def plot_posterior(self):
        return az.plot_trace(self.idata)

    def save(self, path: str):
        self.idata.to_netcdf(path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "BayesianMMM":
        obj = cls(**kwargs)
        obj.idata = az.from_netcdf(path)
        return obj

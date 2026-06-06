"""Budget optimizer: maximize predicted KPI subject to a total impressions constraint.

Only controllable channels are optimized. Fixed channels (organic, affiliate) are
held constant — their MMM contributions are real but we have no direct lever to
pull on them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Channels we can directly buy / control volume on
CONTROLLABLE_COLS = [
    "Paid_Views",           # YouTube paid ads
    "Google_Impressions",   # Google Ads budget
    "Email_Impressions",    # Emails deployed
    "Facebook_Impressions", # Facebook Ads budget
]

# Channels included in the MMM but excluded from optimisation
FIXED_COLS = [
    "Organic_Views",          # YouTube organic — driven by content, not spend
    "Affiliate_Impressions",  # Performance-based — can't directly purchase impressions
]


class BudgetOptimizer:
    """
    Gradient-based budget optimizer using response-curve parameters from a
    fitted BayesianMMM instance.

    Optimises over CONTROLLABLE_COLS only.
    FIXED_COLS are modelled but their volumes are frozen at current levels.

    The optimizer maximises  Σ_c  saturation(adstock(impressions_c))
    subject to  Σ_c impressions_c == total_budget  and  lb_c <= impressions_c <= ub_c.

    Note: "budget" here means total impression/view volume, not dollar spend,
    because the dataset contains impressions rather than currency amounts.
    """

    def __init__(self, mmm, controllable_cols: list[str] | None = None):
        self.mmm = mmm
        self.controllable_cols = controllable_cols or CONTROLLABLE_COLS
        # Validate every controllable col is actually in the fitted model
        unknown = set(self.controllable_cols) - set(self.mmm.channel_cols)
        if unknown:
            raise ValueError(f"Controllable cols not in model: {unknown}")

    def _response(self, impressions: np.ndarray) -> float:
        """Predicted total contribution for controllable channels (negative for minimisation)."""
        total = 0.0
        for i, col in enumerate(self.controllable_cols):
            alpha_mean = float(
                self.mmm.idata.posterior[f"adstock_{col}_alpha"].mean()
            )
            k_mean = float(
                self.mmm.idata.posterior[f"saturation_{col}_lam"].mean()
            )
            x = impressions[i] * (1 / (1 - alpha_mean + 1e-6))
            total += x / (k_mean + x)
        return -total

    def optimize(
        self,
        total_budget: float,
        lower_bounds: list[float] | None = None,
        upper_bounds: list[float] | None = None,
        current_volumes: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        total_budget    : total impression/view volume to allocate across controllable channels
        lower_bounds    : min impressions per channel (default 0)
        upper_bounds    : max impressions per channel (default total_budget)
        current_volumes : current mean impressions per channel (used as x0 and for delta)
        """
        n = len(self.controllable_cols)
        lb = lower_bounds if lower_bounds is not None else [0.0] * n
        ub = upper_bounds if upper_bounds is not None else [total_budget] * n
        x0 = current_volumes if current_volumes is not None else np.full(n, total_budget / n)

        constraints = {"type": "eq", "fun": lambda x: x.sum() - total_budget}
        bounds = list(zip(lb, ub))

        result = minimize(
            self._response,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 500},
        )

        return pd.DataFrame(
            {
                "channel": self.controllable_cols,
                "current_volume": x0,
                "optimal_volume": result.x,
                "delta": result.x - x0,
                "pct_change": (result.x - x0) / (x0 + 1e-9) * 100,
            }
        )

    def roas_curve(
        self, channel: str, volume_range: np.ndarray | None = None
    ) -> pd.DataFrame:
        """
        Spend-response curve for a single controllable channel.
        Raises if a non-controllable channel is requested.
        """
        if channel not in self.controllable_cols:
            raise ValueError(
                f"'{channel}' is not a controllable channel. "
                f"Response curves are only meaningful for: {self.controllable_cols}"
            )
        if volume_range is None:
            volume_range = np.linspace(0, 1e6, 200)

        alpha_mean = float(
            self.mmm.idata.posterior[f"adstock_{channel}_alpha"].mean()
        )
        k_mean = float(
            self.mmm.idata.posterior[f"saturation_{channel}_lam"].mean()
        )

        responses = [
            s * (1 / (1 - alpha_mean + 1e-6)) / (k_mean + s * (1 / (1 - alpha_mean + 1e-6)))
            for s in volume_range
        ]

        return pd.DataFrame({"volume": volume_range, "response": responses})

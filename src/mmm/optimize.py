"""Budget optimizer: maximize predicted KPI subject to a total spend constraint."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class BudgetOptimizer:
    """
    Gradient-based budget optimizer using response-curve parameters from a
    fitted BayesianMMM instance.

    The optimizer maximises  Σ_c  saturation(adstock(spend_c))
    subject to  Σ_c spend_c == total_budget  and  lb_c <= spend_c <= ub_c.
    """

    def __init__(self, mmm):
        self.mmm = mmm

    def _response(self, spend: np.ndarray) -> float:
        """Predicted total contribution given spend vector (negative for minimisation)."""
        if self.mmm.mmm is None:
            raise RuntimeError("MMM must be fitted before optimisation.")
        # Use the model's built-in prediction on a single-period basis
        total = 0.0
        for i, col in enumerate(self.mmm.channel_cols):
            # Simplified single-period Hill response using posterior means
            alpha_mean = float(
                self.mmm.idata.posterior[f"adstock_{col}_alpha"].mean()
            )
            k_mean = float(
                self.mmm.idata.posterior[f"saturation_{col}_lam"].mean()
            )
            x = spend[i] * (1 / (1 - alpha_mean + 1e-6))  # crude steady-state
            total += x / (k_mean + x)
        return -total  # negative for scipy minimise

    def optimize(
        self,
        total_budget: float,
        lower_bounds: list[float] | None = None,
        upper_bounds: list[float] | None = None,
        current_spend: np.ndarray | None = None,
    ) -> pd.DataFrame:
        n = len(self.mmm.channel_cols)
        lb = lower_bounds if lower_bounds is not None else [0.0] * n
        ub = upper_bounds if upper_bounds is not None else [total_budget] * n
        x0 = current_spend if current_spend is not None else np.full(n, total_budget / n)

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
                "channel": self.mmm.channel_cols,
                "optimal_spend": result.x,
                "current_spend": x0,
                "delta": result.x - x0,
            }
        )

    def roas_curve(
        self, channel: str, spend_range: np.ndarray | None = None
    ) -> pd.DataFrame:
        """Return a spend-vs-response curve for a single channel."""
        if spend_range is None:
            spend_range = np.linspace(0, 1e6, 200)

        i = self.mmm.channel_cols.index(channel)
        alpha_mean = float(
            self.mmm.idata.posterior[f"adstock_{channel}_alpha"].mean()
        )
        k_mean = float(
            self.mmm.idata.posterior[f"saturation_{channel}_lam"].mean()
        )

        responses = []
        for s in spend_range:
            x = s * (1 / (1 - alpha_mean + 1e-6))
            responses.append(x / (k_mean + x))

        return pd.DataFrame({"spend": spend_range, "response": responses})

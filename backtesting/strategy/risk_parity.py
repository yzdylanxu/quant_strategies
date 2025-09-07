import pandas as pd
import numpy as np
from scipy.optimize import minimize


class RiskParityStrategy:
    def __init__(self, lookback: int = 252, rebalance_freq: int = 21):
        self.lookback = lookback
        self.rebalance_freq = rebalance_freq

    def _risk_parity_weights(self, returns_window: pd.DataFrame) -> np.ndarray:
        """Calculate risk parity weights for given returns window"""
        try:
            # Calculate covariance matrix
            cov_matrix = returns_window.cov().values * 252  # Annualized

            n_assets = len(returns_window.columns)

            # Objective: minimize sum of squared risk contribution deviations
            def objective(weights):
                portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
                risk_contrib = weights * (cov_matrix @ weights) / portfolio_vol
                target_contrib = 1.0 / n_assets
                return np.sum((risk_contrib - target_contrib) ** 2)

            # Constraints and bounds
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
            bounds = [(0.01, 0.8) for _ in range(n_assets)]  # 1% to 80% per asset
            initial_guess = np.ones(n_assets) / n_assets

            # Optimize
            result = minimize(objective, initial_guess, method='SLSQP',
                              bounds=bounds, constraints=constraints)

            return result.x if result.success else initial_guess

        except:
            # Fallback to equal weights
            return np.ones(len(returns_window.columns)) / len(returns_window.columns)

    def generate_weights(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Generate portfolio weights over time"""
        # Ensure we have enough data
        if len(returns_data) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} observations")

        # Initialize results
        dates = returns_data.index[self.lookback:]
        weights_df = pd.DataFrame(index=dates, columns=returns_data.columns)

        current_weights = None
        last_rebalance = None

        for date in dates:
            # Check if we need to rebalance
            if (last_rebalance is None or
                    (date - last_rebalance).days >= self.rebalance_freq):
                # Get lookback window
                end_idx = returns_data.index.get_loc(date)
                start_idx = end_idx - self.lookback + 1
                window = returns_data.iloc[start_idx:end_idx + 1]

                # Calculate new weights
                current_weights = self._risk_parity_weights(window)
                last_rebalance = date

            weights_df.loc[date] = current_weights

        return weights_df
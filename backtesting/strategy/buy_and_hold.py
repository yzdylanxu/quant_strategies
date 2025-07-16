import pandas as pd

class BuyAndHoldStrategy:
    """
    Stateless strategy that returns a constant long position.
    """

    def generate_signal(self, prices: pd.Series) -> pd.Series:
        """
        Generate a Series of constant 1.0 signals (long-only).
        """
        return pd.Series(1.0, index=prices.index, name="Signal")
import pandas as pd

class MomentumStrategy:
    def __init__(self, lookback: int):
        self.lookback = lookback

    def generate_signal(self, prices: pd.Series) -> pd.Series:
        rolling_mean = prices.rolling(self.lookback).mean()
        signal = (prices > rolling_mean).astype(int)
        # shift to avoid lookahead bias
        return signal.shift(1).fillna(0)
import pandas as pd

def calculate_position(signal: pd.Series) -> pd.Series:
    return signal.astype(float)

def apply_slippage(returns: pd.Series, slippage_bps: float) -> pd.Series:
    slippage_pct = slippage_bps / 10000
    return returns - slippage_pct * 100  # returns are in %
import pandas as pd

def calculate_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna() * 100

def calculate_volatility(returns: pd.Series) -> float:
    return returns.std() * (252 ** 0.5)
import pandas as pd
import numpy as np

def calculate_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna() * 100

def calculate_volatility(returns: pd.Series) -> float:
    return returns.std() * (252 ** 0.5)

def get_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Index must be a pd.DatetimeIndex")

    daily_rfr = risk_free_rate / 252
    excess_returns = returns / 100 - daily_rfr

    mean_daily = excess_returns.mean()
    std_daily = excess_returns.std()

    sharpe_ratio = (mean_daily / std_daily) * np.sqrt(252)
    return sharpe_ratio

def get_max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns / 100).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def annualize_return(returns: pd.Series) -> float:
    cumulative = (1 + returns / 100).prod()
    n_years = len(returns) / 252
    return (cumulative ** (1 / n_years) - 1) * 100
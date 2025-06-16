from backtesting.valuation.utils import get_sharpe_ratio
import pandas as pd
import numpy as np

def test_sharpe_ratio():
    dates = pd.date_range(start="2022-01-01", periods=252, freq="B")
    returns = pd.Series(np.random.normal(0.05, 1, 252), index=dates)
    sr = get_sharpe_ratio(returns)
    assert isinstance(sr, float)
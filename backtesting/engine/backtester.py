import pandas as pd
from backtesting.execution.executor import calculate_position, apply_slippage
from backtesting.valuation.utils import get_sharpe_ratio, get_max_drawdown, annualize_return, calculate_returns

class BacktestRunner:
    def __init__(self, prices: pd.Series, strategy, slippage_bps: float = 0.0):
        self.prices = prices
        self.strategy = strategy
        self.slippage_bps = slippage_bps

    def run(self) -> dict:
        signal = self.strategy.generate_signal(self.prices)
        returns = calculate_returns(self.prices)
        position = calculate_position(signal)
        strategy_returns = position.shift(1) * returns  # delay by 1 day
        strategy_returns = apply_slippage(strategy_returns, self.slippage_bps)

        metrics = {
            "Annualized Return": float(annualize_return(strategy_returns)),
            "Sharpe Ratio": float(get_sharpe_ratio(strategy_returns)),
            "Max Drawdown": float(get_max_drawdown(strategy_returns)),
        }
        return metrics
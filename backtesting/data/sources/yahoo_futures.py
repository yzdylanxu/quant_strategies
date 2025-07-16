import os
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime

class FuturesInstrument:
    def __init__(self, name: str, ticker: str, data_dir: Path):
        self.name = name
        self.ticker = ticker
        self.data_dir = data_dir
        self.filename = data_dir / f"{name.lower().replace(' ', '_')}.csv"

    def fetch_data(self, start: str = "2000-01-01", end: str = None, force_refresh: bool = True) -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance. Uses local cache if available unless forced.
        """
        if not end:
            end = datetime.today().strftime("%Y-%m-%d")

        # Check if file already exists
        if self.filename.exists() and not force_refresh:
            print(f"[{self.name}] Loading from local cache...")
            return pd.read_csv(self.filename, parse_dates=["Date"], index_col="Date")

        print(f"[{self.name}] Downloading from Yahoo Finance...")
        data = yf.download(self.ticker, start=start, end=end)
        if data.empty:
            raise ValueError(f"No data fetched for {self.name} ({self.ticker})")

        # Clean it up to keep only one price column
        clean_data = pd.DataFrame(index=data.index)
        clean_data["Price"] = data["Close"]
        clean_data.index.name = "Date"

        # Save to CSV cleanly
        clean_data.to_csv(self.filename)
        return clean_data

    def get_data(self) -> pd.DataFrame:
        """
        Load saved data from local CSV.
        """
        if not self.filename.exists():
            raise FileNotFoundError(f"No cached data found for {self.name}")
        return pd.read_csv(self.filename, parse_dates=["Date"], index_col="Date")

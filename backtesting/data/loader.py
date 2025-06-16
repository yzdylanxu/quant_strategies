import pandas as pd

def load_csv_data(filepath: str) -> pd.Series:
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    if "Close" not in df.columns:
        raise ValueError("CSV must have a 'Close' column")
    return df["Close"].sort_index()
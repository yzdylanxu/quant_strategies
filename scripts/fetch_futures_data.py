from datetime import datetime
from pathlib import Path
from backtesting.data.sources.yahoo_futures import FuturesInstrument

# Directory to store raw data
data_dir = Path("../backtesting/data/raw")
data_dir.mkdir(parents=True, exist_ok=True)

# Define futures contracts
instruments = [
    FuturesInstrument("S&P 500 E-Mini", "ES=F", data_dir),
    FuturesInstrument("US 10Y Note", "ZN=F", data_dir),
    FuturesInstrument("Stoxx 50", "FESX.EX", data_dir),
    FuturesInstrument("German Bund", "BUNL", data_dir)
]

# Fetch and cache data
for inst in instruments:
    df = inst.fetch_data(start="2015-01-01")
    print(f"{inst.name} loaded with {len(df)} rows.")
    print(df.head())

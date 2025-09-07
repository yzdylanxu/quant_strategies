from pathlib import Path

# --- Bloomberg Loader ---
try:
    from backtesting.data.sources.bbg_futures_loader_chat import (
        BloombergContinuousFutures, InstrumentConfig
    )
    HAS_BBG = True
except ImportError:
    HAS_BBG = False

# --- Yahoo Loader ---
from backtesting.data.sources.yahoo_futures import FuturesInstrument


def fetch_with_bloomberg():
    """Fetch continuous futures series from Bloomberg."""
    instruments = {
        # Cash-settled equity index future -> roll by LTD (no FND)
        "ES": InstrumentConfig(root="ES", bbg_suffix="Index",  roll_on="LTD"),
        # Physically-deliverable Treasury future -> roll by FND
        "TY": InstrumentConfig(root="TY", bbg_suffix="Comdty", roll_on="FND"),
        # Uncomment and adjust once you verify vendor tickers:
        # "FGBL": InstrumentConfig(root="RX", bbg_suffix="Comdty", roll_on="FND"),
        # "FESX": InstrumentConfig(root="SX5E", bbg_suffix="Index", roll_on="LTD"),
    }

    loader = BloombergContinuousFutures(
        instruments=instruments,
        start_date="2015-01-01",
        end_date=None,              # defaults to today
        roll_offset_bdays=3,        # common practice
        save_dir="Metadata"
    )

    series = loader.build_and_save()
    print("Bloomberg data saved under Metadata/")
    return series


def fetch_with_yahoo():
    """Fetch raw futures data from Yahoo Finance."""
    data_dir = Path("backtesting/data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    instruments = [
        FuturesInstrument("S&P 500 E-Mini", "ES=F", data_dir),
        FuturesInstrument("US 10Y Note", "ZN=F", data_dir),
        FuturesInstrument("Stoxx 50", "FESX.EX", data_dir),
        FuturesInstrument("German Bund", "BUNL", data_dir)
    ]

    for inst in instruments:
        df = inst.fetch_data(start="2015-01-01")
        print(f"{inst.name} loaded with {len(df)} rows.")
        print(df.head())


if __name__ == "__main__":
    if HAS_BBG:
        print("Using Bloomberg loader...")
        fetch_with_bloomberg()
    else:
        print("Bloomberg not available, falling back to Yahoo Finance...")
        fetch_with_yahoo()

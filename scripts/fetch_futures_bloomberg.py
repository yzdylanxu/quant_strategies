import sys
import os

# Add the sources directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sources_dir = os.path.join(project_root, 'backtesting', 'data', 'sources')
sys.path.append(sources_dir)

from backtesting.data.sources.bbg_futures_loader import BloombergFuturesLoader


def main():
    print("Fetching Bloomberg futures data...")

    # Initialize loader
    loader = BloombergFuturesLoader()

    try:
        # Load ES and TY data from 2015 to now
        loader.load_all_instruments(start_date="2015-01-01")
        print("✓ Data fetch completed successfully!")

    except Exception as e:
        print(f"✗ Error: {e}")

    finally:
        # Clean up
        loader.cleanup()


if __name__ == "__main__":
    main()
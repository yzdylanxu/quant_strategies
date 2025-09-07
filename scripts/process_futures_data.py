#!/usr/bin/env python3
"""
Process Futures Data Script

This script loads Bloomberg futures data from Excel files and creates continuous
price series with proper rolling logic for ES and TY contracts.

Usage: python scripts/process_futures_data.py
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "backtesting" / "data" / "sources"))


def main(excel_futures_loader=None):
    """Main function to process futures data"""

    print("=" * 60)
    print("FUTURES DATA PROCESSING")
    print("=" * 60)

    try:
        # Import the Excel futures loader
        from excel_futures_loader import ExcelFuturesLoader
        print("✓ Excel futures loader imported successfully")

        # Check if required files exist
        raw_data_dir = project_root / "backtesting" / "data" / "raw"
        processed_data_dir = project_root / "backtesting" / "data" / "processed"
        metadata_dir = project_root / "backtesting" / "data" / "metadata"

        price_files = ["ES_Prices.xlsx", "TY_Prices.xlsx"]
        metadata_files = ["metadata.xlsx"]

        print(f"\nChecking for price files in: {raw_data_dir}")
        missing_files = []
        for file in price_files:
            file_path = raw_data_dir / file
            if file_path.exists():
                print(f"✓ Found {file}")
            else:
                print(f"✗ Missing {file}")
                missing_files.append(f"raw/{file}")

        print(f"\nChecking for metadata files in: {metadata_dir}")
        for file in metadata_files:
            file_path = metadata_dir / file
            if file_path.exists():
                print(f"✓ Found {file}")
            else:
                print(f"✗ Missing {file}")
                missing_files.append(f"metadata/{file}")

        if missing_files:
            print(f"\nError: Missing required files: {missing_files}")
            print("Please ensure files are in the correct directories:")
            print("  - Price files: backtesting/data/raw/")
            print("  - Metadata files: backtesting/data/metadata/")
            return False

        # Create directories if they don't exist
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Processed data directory: {processed_data_dir}")

        # Initialize the loader
        print(f"\nInitializing Excel Futures Loader...")
        loader = ExcelFuturesLoader(
            raw_data_dir=str(raw_data_dir),
            processed_data_dir=str(processed_data_dir),
            metadata_dir=str(metadata_dir)
        )

        # Process the data
        print(f"\nProcessing futures data (2016-01-01 to present)...")
        loader.load_all_instruments(
            start_date="2016-01-01"
        )

        print("\n" + "=" * 60)
        print("DATA PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        print(f"\nRaw data location: {raw_data_dir}")
        print("- metadata.xlsx")
        print("- ES_Prices.xlsx")
        print("- TY_Prices.xlsx")

        print(f"\nProcessed data location: {processed_data_dir}")
        print("- ES_continuous.csv (S&P 500 E-mini continuous series)")
        print("- TY_continuous.csv (10Y Treasury continuous series)")
        print("- ES_summary.csv (ES performance statistics)")
        print("- TY_summary.csv (TY performance statistics)")

        return True

    except Exception as e:
        print(f"✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pickle
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import warnings

try:
    import blpapi
except ImportError:
    print("Bloomberg API not available. Please install blpapi package.")
    blpapi = None


@dataclass
class FuturesSpec:
    """Configuration for a specific futures contract"""
    name: str
    bbg_ticker_base: str  # e.g., "ES" or "TY"
    bbg_ticker_suffix: str  # e.g., "Index" or "Comdty"
    roll_rule: str  # "LTD" or "FND"
    roll_days_before: int = 3
    value_of_1pt_field: str = "VALUE_OF_1PT"
    settlement_field: str = "PX_SETTLE"


class BloombergFuturesLoader:
    """
    Bloomberg Futures Data Loader with automatic contract rolling and continuous series generation.
    Handles contract expiry, roll dates, and creates continuous price series.
    """

    def __init__(self, cache_dir: str = None):
        """
        Initialize the Bloomberg Futures Loader

        Args:
            cache_dir: Directory to store cached data (default: relative to this file)
        """
        if cache_dir is None:
            # Default to metadata/cache relative to the backtesting/data directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.dirname(current_dir)  # Go up to backtesting/data/
            cache_dir = os.path.join(data_dir, "metadata", "cache")
        """
        Initialize the Bloomberg Futures Loader

        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Contract month codes
        self.month_codes = {
            3: 'H',  # March
            6: 'M',  # June
            9: 'U',  # September
            12: 'Z'  # December
        }

        # Bloomberg session
        self.session = None
        self._initialize_bloomberg()

        # Predefined futures specifications
        self.futures_specs = {
            'ES': FuturesSpec(
                name="S&P_500_E-Mini",
                bbg_ticker_base="ES",
                bbg_ticker_suffix="Index",
                roll_rule="LTD",
                roll_days_before=3
            ),
            'TY': FuturesSpec(
                name="US_10Y_Note",
                bbg_ticker_base="TY",
                bbg_ticker_suffix="Comdty",
                roll_rule="FND",
                roll_days_before=3
            ),
            'BUND': FuturesSpec(
                name="German_Bund",
                bbg_ticker_base="RX",
                bbg_ticker_suffix="Comdty",
                roll_rule="FND",
                roll_days_before=3
            ),
            'STOXX': FuturesSpec(
                name="STOXX_50",
                bbg_ticker_base="VG",
                bbg_ticker_suffix="Index",
                roll_rule="LTD",
                roll_days_before=3
            )
        }

    def _initialize_bloomberg(self):
        """Initialize Bloomberg API connection"""
        if blpapi is None:
            self.logger.warning("Bloomberg API not available. Running in mock mode.")
            return

        try:
            # Start Bloomberg session
            sessionOptions = blpapi.SessionOptions()
            sessionOptions.setServerHost("localhost")
            sessionOptions.setServerPort(8194)

            self.session = blpapi.Session(sessionOptions)

            if not self.session.start():
                raise Exception("Failed to start Bloomberg session")

            if not self.session.openService("//blp/refdata"):
                raise Exception("Failed to open Bloomberg reference data service")

            self.logger.info("Bloomberg session initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Bloomberg: {e}")
            self.session = None

    def generate_contract_tickers(self, base_ticker: str, suffix: str,
                                  start_year: int = 2015, end_year: Optional[int] = None) -> List[str]:
        """
        Generate list of quarterly futures contract tickers

        Args:
            base_ticker: Base ticker (e.g., 'ES', 'TY')
            suffix: Bloomberg suffix ('Index' or 'Comdty')
            start_year: Start year for contract generation
            end_year: End year (default: current year + 1)

        Returns:
            List of Bloomberg tickers
        """
        if end_year is None:
            end_year = datetime.now().year + 1

        tickers = []
        for year in range(start_year, end_year + 1):
            year_suffix = str(year)[-1]  # Last digit of year
            for month in [3, 6, 9, 12]:  # Quarterly contracts
                month_code = self.month_codes[month]
                ticker = f"{base_ticker}{month_code}{year_suffix} {suffix}"
                tickers.append(ticker)

        return tickers

    def _get_cache_filename(self, ticker: str, data_type: str) -> str:
        """Generate cache filename for a ticker and data type"""
        safe_ticker = ticker.replace(" ", "_").replace("/", "_")
        return os.path.join(self.cache_dir, f"{safe_ticker}_{data_type}.pkl")

    def _load_from_cache(self, ticker: str, data_type: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available"""
        cache_file = self._get_cache_filename(ticker, data_type)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache for {ticker}: {e}")
        return None

    def _save_to_cache(self, ticker: str, data_type: str, data: pd.DataFrame):
        """Save data to cache"""
        cache_file = self._get_cache_filename(ticker, data_type)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.warning(f"Failed to save cache for {ticker}: {e}")

    def fetch_settlement_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch settlement prices for a specific contract

        Args:
            ticker: Bloomberg ticker (e.g., "ESU5 Index")
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with dates and settlement prices
        """
        # Check cache first
        cached_data = self._load_from_cache(ticker, "settlement")
        if cached_data is not None:
            # Filter cached data for requested date range
            mask = (cached_data.index >= start_date) & (cached_data.index <= end_date)
            if mask.any():
                self.logger.info(f"Using cached settlement data for {ticker}")
                return cached_data[mask]

        if self.session is None:
            self.logger.warning(f"Bloomberg not available. Returning mock data for {ticker}")
            # Return mock data for testing
            dates = pd.date_range(start_date, end_date, freq='B')  # Business days
            mock_prices = np.random.normal(4000, 50, len(dates))  # Mock S&P prices
            df = pd.DataFrame({'PX_SETTLE': mock_prices}, index=dates)
            return df

        try:
            # Create Bloomberg request
            refDataService = self.session.getService("//blp/refdata")
            request = refDataService.createRequest("HistoricalDataRequest")
            request.append("securities", ticker)
            request.append("fields", "PX_SETTLE")
            request.set("startDate", start_date.replace("-", ""))
            request.set("endDate", end_date.replace("-", ""))

            # Send request
            self.session.sendRequest(request)

            # Process response
            data = []
            while True:
                event = self.session.nextEvent(500)
                for msg in event:
                    if msg.hasElement("securityData"):
                        securityData = msg.getElement("securityData")
                        fieldData = securityData.getElement("fieldData")

                        for i in range(fieldData.numValues()):
                            point = fieldData.getValueAsElement(i)
                            date = point.getElement("date").getValue()
                            price = point.getElement("PX_SETTLE").getValue()
                            data.append((date, price))

                if event.eventType() == blpapi.Event.RESPONSE:
                    break

            # Create DataFrame
            df = pd.DataFrame(data, columns=['date', 'PX_SETTLE'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            # Cache the data
            self._save_to_cache(ticker, "settlement", df)

            self.logger.info(f"Fetched {len(df)} settlement prices for {ticker}")
            return df

        except Exception as e:
            self.logger.error(f"Failed to fetch settlement prices for {ticker}: {e}")
            return pd.DataFrame()

    def fetch_contract_info(self, ticker: str) -> Dict:
        """
        Fetch contract information (last trade date, first notice date, value of 1pt)

        Args:
            ticker: Bloomberg ticker

        Returns:
            Dictionary with contract information
        """
        # Check cache
        cached_info = self._load_from_cache(ticker, "info")
        if cached_info is not None and len(cached_info) > 0:
            self.logger.info(f"Using cached contract info for {ticker}")
            return cached_info.iloc[0].to_dict()

        if self.session is None:
            # Return mock data
            mock_info = {
                'LAST_TRADEABLE_DT': datetime(2025, 9, 15),  # 3rd Friday of September
                'FUT_NOTICE_FIRST': datetime(2025, 9, 1),
                'VALUE_OF_1PT': 50.0  # $50 per point for ES
            }
            self.logger.warning(f"Bloomberg not available. Using mock contract info for {ticker}")
            return mock_info

        try:
            # Create Bloomberg request
            refDataService = self.session.getService("//blp/refdata")
            request = refDataService.createRequest("ReferenceDataRequest")
            request.append("securities", ticker)
            request.append("fields", "LAST_TRADEABLE_DT")
            request.append("fields", "FUT_NOTICE_FIRST")
            request.append("fields", "VALUE_OF_1PT")

            # Send request
            self.session.sendRequest(request)

            # Process response
            info = {}
            while True:
                event = self.session.nextEvent(500)
                for msg in event:
                    if msg.hasElement("securityData"):
                        securityData = msg.getElement("securityData")
                        fieldData = securityData.getElement("fieldData")

                        for i in range(fieldData.numElements()):
                            field = fieldData.getElement(i)
                            field_name = str(field.name())
                            field_value = field.getValue()
                            info[field_name] = field_value

                if event.eventType() == blpapi.Event.RESPONSE:
                    break

            # Cache the info
            info_df = pd.DataFrame([info])
            self._save_to_cache(ticker, "info", info_df)

            self.logger.info(f"Fetched contract info for {ticker}")
            return info

        except Exception as e:
            self.logger.error(f"Failed to fetch contract info for {ticker}: {e}")
            return {}

    def calculate_roll_dates(self, contracts_info: Dict[str, Dict]) -> pd.DataFrame:
        """
        Calculate roll dates for all contracts based on their specifications

        Args:
            contracts_info: Dictionary mapping tickers to their contract info

        Returns:
            DataFrame with roll schedule
        """
        roll_schedule = []

        for ticker, info in contracts_info.items():
            # Determine which date to use for rolling
            if 'LAST_TRADEABLE_DT' in info and info['LAST_TRADEABLE_DT']:
                expiry_date = pd.to_datetime(info['LAST_TRADEABLE_DT'])
                roll_type = 'LTD'
            elif 'FUT_NOTICE_FIRST' in info and info['FUT_NOTICE_FIRST']:
                expiry_date = pd.to_datetime(info['FUT_NOTICE_FIRST'])
                roll_type = 'FND'
            else:
                continue

            # Calculate roll date (3 days before)
            roll_date = expiry_date - timedelta(days=3)

            # Adjust for weekends (move to previous business day)
            while roll_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                roll_date -= timedelta(days=1)

            roll_schedule.append({
                'ticker': ticker,
                'expiry_date': expiry_date,
                'roll_date': roll_date,
                'roll_type': roll_type,
                'value_of_1pt': info.get('VALUE_OF_1PT', 1.0)
            })

        return pd.DataFrame(roll_schedule).sort_values('roll_date')

    def create_continuous_series(self, spec: FuturesSpec, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Create continuous futures series with roll adjustments

        Args:
            spec: FuturesSpec configuration
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with continuous price series
        """
        self.logger.info(f"Creating continuous series for {spec.name}")

        # Generate all contract tickers
        tickers = self.generate_contract_tickers(
            spec.bbg_ticker_base,
            spec.bbg_ticker_suffix,
            start_year=2015
        )

        # Fetch contract information for all tickers
        contracts_info = {}
        for ticker in tickers:
            info = self.fetch_contract_info(ticker)
            if info:
                contracts_info[ticker] = info

        # Calculate roll schedule
        roll_schedule = self.calculate_roll_dates(contracts_info)

        if roll_schedule.empty:
            self.logger.error(f"No valid contracts found for {spec.name}")
            return pd.DataFrame()

        # Filter roll schedule to our date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        relevant_contracts = roll_schedule[
            (roll_schedule['roll_date'] >= start_dt - timedelta(days=365)) &
            (roll_schedule['roll_date'] <= end_dt + timedelta(days=365))
            ].copy()

        # Fetch price data for relevant contracts
        price_data = {}
        for _, contract in relevant_contracts.iterrows():
            ticker = contract['ticker']

            # Determine date range for this contract
            contract_start = max(start_dt - timedelta(days=30),
                                 pd.to_datetime('2015-01-01'))
            contract_end = min(contract['expiry_date'] + timedelta(days=30),
                               end_dt + timedelta(days=30))

            prices = self.fetch_settlement_prices(
                ticker,
                contract_start.strftime('%Y-%m-%d'),
                contract_end.strftime('%Y-%m-%d')
            )

            if not prices.empty:
                price_data[ticker] = prices

        if not price_data:
            self.logger.error(f"No price data found for {spec.name}")
            return pd.DataFrame()

        # Build continuous series
        continuous_data = []
        current_contract_idx = 0

        # Start with the first contract that has data
        business_days = pd.bdate_range(start_date, end_date)

        for date in business_days:
            # Check if we need to roll to next contract
            while (current_contract_idx < len(relevant_contracts) - 1 and
                   date >= relevant_contracts.iloc[current_contract_idx]['roll_date']):
                current_contract_idx += 1

            # Get current contract
            if current_contract_idx >= len(relevant_contracts):
                break

            current_contract = relevant_contracts.iloc[current_contract_idx]
            current_ticker = current_contract['ticker']

            # Get price for this date
            if current_ticker in price_data:
                contract_prices = price_data[current_ticker]
                if date in contract_prices.index:
                    price = contract_prices.loc[date, 'PX_SETTLE']

                    continuous_data.append({
                        'date': date,
                        'raw_contract': current_ticker,
                        'raw_price': price,
                        'continuous_price': price,  # Will be adjusted later
                        'value_of_1pt': current_contract['value_of_1pt']
                    })

        # Convert to DataFrame
        df = pd.DataFrame(continuous_data)
        if df.empty:
            return df

        df.set_index('date', inplace=True)

        # Apply roll adjustments (back-adjust method)
        df['continuous_price'] = df['raw_price'].copy()
        df['roll_adjustment'] = 0.0

        # Apply adjustments backwards from most recent
        for i in range(len(relevant_contracts) - 2, -1, -1):
            roll_date = relevant_contracts.iloc[i]['roll_date']
            next_roll_date = relevant_contracts.iloc[i + 1]['roll_date']

            # Find the price gap at roll
            roll_mask = df.index == roll_date
            if roll_mask.any():
                current_price = df.loc[roll_date, 'raw_price']

                # Find next contract's price on roll date
                next_ticker = relevant_contracts.iloc[i + 1]['ticker']
                if next_ticker in price_data:
                    next_prices = price_data[next_ticker]
                    if roll_date in next_prices.index:
                        next_price = next_prices.loc[roll_date, 'PX_SETTLE']
                        adjustment = current_price - next_price

                        # Apply adjustment to all dates before roll
                        before_roll = df.index < roll_date
                        df.loc[before_roll, 'continuous_price'] += adjustment
                        df.loc[before_roll, 'roll_adjustment'] += adjustment

        # Calculate returns
        df['returns'] = df['continuous_price'].pct_change()
        df['log_returns'] = np.log(df['continuous_price'] / df['continuous_price'].shift(1))

        self.logger.info(f"Created continuous series for {spec.name} with {len(df)} observations")
        return df

    def export_to_csv(self, instrument: str, start_date: str = "2015-01-01",
                      end_date: Optional[str] = None, output_dir: str = None):
        """
        Export continuous futures series to CSV

        Args:
            instrument: Instrument code (e.g., 'ES', 'TY')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date (default: today)
            output_dir: Output directory for CSV files (default: ../metadata relative to this file)
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        if output_dir is None:
            # Default to metadata folder relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.dirname(current_dir)  # Go up to backtesting/data/
            output_dir = os.path.join(data_dir, "metadata")

        if instrument not in self.futures_specs:
            self.logger.error(f"Unknown instrument: {instrument}")
            return

        spec = self.futures_specs[instrument]

        # Create continuous series
        df = self.create_continuous_series(spec, start_date, end_date)

        if df.empty:
            self.logger.error(f"No data to export for {instrument}")
            return

        # Prepare output directory
        os.makedirs(output_dir, exist_ok=True)

        # Export to CSV
        filename = f"{spec.name.lower()}.csv"
        filepath = os.path.join(output_dir, filename)

        df.to_csv(filepath)
        self.logger.info(f"Exported {instrument} data to {filepath}")

        # Also save summary statistics
        summary_filename = f"{spec.name.lower()}_summary.csv"
        summary_filepath = os.path.join(output_dir, summary_filename)

        summary_stats = df[['continuous_price', 'returns', 'log_returns']].describe()
        summary_stats.to_csv(summary_filepath)

        self.logger.info(f"Exported summary statistics to {summary_filepath}")

    def load_all_instruments(self, start_date: str = "2015-01-01",
                             end_date: Optional[str] = None, output_dir: str = None):
        """
        Load and export all configured instruments

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date (default: today)
            output_dir: Output directory for CSV files (default: ../metadata relative to this file)
        """
        if output_dir is None:
            # Default to metadata folder relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.dirname(current_dir)  # Go up to backtesting/data/
            output_dir = os.path.join(data_dir, "metadata")
        instruments = ['ES', 'TY']  # Start with these two as requested

        for instrument in instruments:
            try:
                self.logger.info(f"Processing {instrument}...")
                self.export_to_csv(instrument, start_date, end_date, output_dir)
            except Exception as e:
                self.logger.error(f"Failed to process {instrument}: {e}")

    def cleanup(self):
        """Clean up Bloomberg session"""
        if self.session:
            self.session.stop()


# Example usage and main execution
if __name__ == "__main__":
    # Initialize the loader
    loader = BloombergFuturesLoader(cache_dir="metadata/cache")

    try:
        # Load S&P 500 E-Mini and US 10Y Note data
        loader.load_all_instruments(
            start_date="2015-01-01",
            output_dir="metadata"
        )

        print("Data loading completed successfully!")
        print("Check the 'metadata' folder for CSV files:")
        print("- s&p_500_e-mini.csv")
        print("- us_10y_note.csv")
        print("- Summary statistics files")

    except Exception as e:
        print(f"Error during data loading: {e}")

    finally:
        # Clean up
        loader.cleanup()
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from dataclasses import dataclass

from backtesting.valuation.utils import annualize_return, calculate_volatility, get_sharpe_ratio, get_max_drawdown


@dataclass
class FuturesSpec:
    """Configuration for a specific futures contract"""
    name: str
    bbg_ticker_base: str  # e.g., "ES" or "TY"
    bbg_ticker_suffix: str  # e.g., "Index" or "Comdty"
    roll_rule: str  # "LTD" or "FND"
    roll_days_before: int = 3
    value_of_1pt: float = 50.0


class ExcelFuturesLoader:
    """
    Excel Futures Data Loader - reads Bloomberg data from Excel files
    and creates continuous series with proper rolling logic
    """

    def __init__(self, raw_data_dir: str = None, processed_data_dir: str = None, metadata_dir: str = None):
        """
        Initialize the Excel Futures Loader

        Args:
            raw_data_dir: Directory containing raw Excel price files (default: backtesting/data/raw)
            processed_data_dir: Directory for processed output files (default: backtesting/data/processed)
            metadata_dir: Directory containing metadata files (default: backtesting/data/metadata)
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up to quant_strategies

        if raw_data_dir is None:
            raw_data_dir = os.path.join(project_root, "backtesting", "data", "raw")

        if processed_data_dir is None:
            processed_data_dir = os.path.join(project_root, "backtesting", "data", "processed")

        if metadata_dir is None:
            metadata_dir = os.path.join(project_root, "backtesting", "data", "metadata")

        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.metadata_dir = metadata_dir

        # Create directories if they don't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

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

        # Predefined futures specifications
        self.futures_specs = {
            'ES': FuturesSpec(
                name="S&P_500_E-Mini",
                bbg_ticker_base="ES",
                bbg_ticker_suffix="Index",
                roll_rule="LTD",
                roll_days_before=3,
                value_of_1pt=50.0
            ),
            'TY': FuturesSpec(
                name="US_10Y_Note",
                bbg_ticker_base="TY",
                bbg_ticker_suffix="Comdty",
                roll_rule="FND",
                roll_days_before=3,
                value_of_1pt=1000.0
            )
        }

        # Load metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> pd.DataFrame:
        """Load contract metadata from Excel file"""
        metadata_path = os.path.join(self.metadata_dir, "metadata.xlsx")
        try:
            df = pd.read_excel(metadata_path)
            # Convert date columns to datetime
            date_columns = ['Last_Trade_Date', 'First_Notice_Date', 'Start_Date', 'End_Date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            self.logger.info(f"Loaded metadata for {len(df)} contracts")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return pd.DataFrame()

    def _load_contract_prices(self, ticker: str, instrument: str) -> pd.DataFrame:
        """Load price data for a specific contract from Excel"""
        # Determine which Excel file to use
        if instrument == 'ES':
            excel_file = "ES_Prices.xlsx"
        elif instrument == 'TY':
            excel_file = "TY_Prices.xlsx"
        else:
            raise ValueError(f"Unknown instrument: {instrument}")

        excel_path = os.path.join(self.data_dir, excel_file)

        try:
            # Extract contract name from ticker (e.g., "ESH25 Index" -> "ESH25")
            contract_name = ticker.split()[0]

            # Load the specific worksheet
            df = pd.read_excel(excel_path, sheet_name=contract_name)

            # Clean up the data
            if len(df.columns) >= 2:
                df.columns = ['Date', 'PX_SETTLE']
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                df = df.dropna()
                df.sort_index(inplace=True)

                self.logger.info(f"Loaded {len(df)} prices for {ticker}")
                return df
            else:
                self.logger.warning(f"Insufficient data columns for {ticker}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Failed to load prices for {ticker}: {e}")
            return pd.DataFrame()

    def _get_roll_date(self, contract_info: pd.Series, spec: FuturesSpec) -> datetime:
        """Calculate roll date based on contract specification"""
        if spec.roll_rule == "LTD" and 'Last_Trade_Date' in contract_info:
            expiry_date = contract_info['Last_Trade_Date']
        elif spec.roll_rule == "FND" and 'First_Notice_Date' in contract_info:
            expiry_date = contract_info['First_Notice_Date']
        else:
            # Fallback
            expiry_date = contract_info.get('Last_Trade_Date', contract_info.get('First_Notice_Date'))

        if pd.isna(expiry_date):
            return None

        # Calculate roll date (3 days before expiry)
        roll_date = expiry_date - timedelta(days=spec.roll_days_before)

        # Adjust for weekends (move to previous business day)
        while roll_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            roll_date -= timedelta(days=1)

        return roll_date

    def create_continuous_series(self, instrument: str, start_date: str = "2016-01-01",
                                 end_date: str = None) -> pd.DataFrame:
        """
        Create continuous futures series with roll adjustments

        Args:
            instrument: 'ES' or 'TY'
            start_date: Start date in YYYY-MM-DD format
            end_date: End date (default: today)

        Returns:
            DataFrame with continuous price series including rx_diff and rx_pct
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        if instrument not in self.futures_specs:
            raise ValueError(f"Unknown instrument: {instrument}")

        spec = self.futures_specs[instrument]
        self.logger.info(f"Creating continuous series for {spec.name}")

        # Filter metadata for this instrument
        instrument_metadata = self.metadata[
            self.metadata['Ticker'].str.startswith(
                f"{instrument}{spec.bbg_ticker_base[2:]}" if len(spec.bbg_ticker_base) > 2 else instrument)
        ].copy()

        if instrument_metadata.empty:
            self.logger.error(f"No metadata found for {instrument}")
            return pd.DataFrame()

        # Calculate roll dates
        instrument_metadata['roll_date'] = instrument_metadata.apply(
            lambda row: self._get_roll_date(row, spec), axis=1
        )

        # Filter contracts for date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        relevant_contracts = instrument_metadata[
            (instrument_metadata['roll_date'] >= start_dt - timedelta(days=365)) &
            (instrument_metadata['roll_date'] <= end_dt + timedelta(days=365))
            ].sort_values('roll_date').reset_index(drop=True)

        if relevant_contracts.empty:
            self.logger.error(f"No relevant contracts found for {instrument} in date range")
            return pd.DataFrame()

        # Load price data for all relevant contracts
        price_data = {}
        for _, contract in relevant_contracts.iterrows():
            ticker = contract['Ticker']
            prices = self._load_contract_prices(ticker, instrument)
            if not prices.empty:
                price_data[ticker] = prices

        if not price_data:
            self.logger.error(f"No price data loaded for {instrument}")
            return pd.DataFrame()

        # Build continuous series with proper rolling logic
        continuous_data = []
        business_days = pd.bdate_range(start_date, end_date)

        current_contract_idx = 0
        previous_close = None

        for i, date in enumerate(business_days):
            # Check if we need to roll to next contract
            is_roll_day = False
            if (current_contract_idx < len(relevant_contracts) - 1 and
                    date >= relevant_contracts.iloc[current_contract_idx]['roll_date']):
                is_roll_day = True
                current_contract_idx += 1

            # Get current contract
            if current_contract_idx >= len(relevant_contracts):
                break

            current_contract = relevant_contracts.iloc[current_contract_idx]
            current_ticker = current_contract['Ticker']

            # Get price for this date
            if current_ticker in price_data:
                contract_prices = price_data[current_ticker]
                if date in contract_prices.index:
                    current_close = contract_prices.loc[date, 'PX_SETTLE']

                    # Calculate rx_diff and rx_pct
                    if i == 0:  # First day
                        rx_diff = 0.0
                        rx_pct = 0.0
                    elif is_roll_day:
                        # Roll day: new contract close - yesterday's old contract close
                        if previous_close is not None:
                            rx_diff = current_close - previous_close
                            rx_pct = rx_diff / previous_close if previous_close != 0 else 0.0
                        else:
                            rx_diff = 0.0
                            rx_pct = 0.0
                    else:
                        # Normal day: today's close - yesterday's close (same contract)
                        if previous_close is not None:
                            rx_diff = current_close - previous_close
                            rx_pct = rx_diff / previous_close if previous_close != 0 else 0.0
                        else:
                            rx_diff = 0.0
                            rx_pct = 0.0

                    continuous_data.append({
                        'date': date,
                        'contract_ticker': current_ticker,
                        'closing_price': current_close,
                        'rx_diff': rx_diff,
                        'rx_pct': rx_pct,
                        'value_of_1pt': spec.value_of_1pt,
                        'is_roll_day': is_roll_day
                    })

                    # Update previous close for next iteration
                    previous_close = current_close

        # Convert to DataFrame
        if not continuous_data:
            self.logger.error(f"No continuous data created for {instrument}")
            return pd.DataFrame()

        df = pd.DataFrame(continuous_data)
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        # Calculate additional metrics
        df['log_returns'] = np.log(df['closing_price'] / df['closing_price'].shift(1))
        df['cumulative_returns'] = (1 + df['rx_pct']).cumprod() - 1

        # Calculate rolling volatility (21-day)
        df['volatility_21d'] = df['rx_pct'].rolling(window=21).std() * np.sqrt(252)

        self.logger.info(f"Created continuous series for {instrument} with {len(df)} observations")
        return df

    def export_continuous_series(self, instrument: str, output_dir: str = None,
                                 start_date: str = "2016-01-01", end_date: str = None):
        """Export continuous series to CSV"""
        if output_dir is None:
            output_dir = self.data_dir

        df = self.create_continuous_series(instrument, start_date, end_date)

        if df.empty:
            self.logger.error(f"No data to export for {instrument}")
            return

        # Export main data
        filename = f"{instrument}_continuous.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath)

        # Calculate performance metrics using existing utility functions
        # Convert rx_pct to percentage format for utility functions
        returns_pct = df['rx_pct'] * 100  # Convert decimal to percentage
        prices = df['closing_price']

        try:
            # Use your existing functions
            total_return_annualized = annualize_return(returns_pct)
            volatility = calculate_volatility(returns_pct)
            sharpe_ratio = get_sharpe_ratio(returns_pct, risk_free_rate=0.02)  # Assuming 2% risk-free rate
            max_drawdown = get_max_drawdown(returns_pct)

            # Additional calculations
            total_return_cumulative = (1 + df['rx_pct']).prod() - 1

            summary_stats = {
                'Instrument': instrument,
                'Start_Date': df.index[0].strftime('%Y-%m-%d'),
                'End_Date': df.index[-1].strftime('%Y-%m-%d'),
                'Total_Return_Cumulative': total_return_cumulative,
                'Total_Return_Annualized_Pct': total_return_annualized,
                'Annualized_Volatility_Pct': volatility,
                'Sharpe_Ratio': sharpe_ratio,
                'Max_Drawdown_Pct': abs(max_drawdown * 100),  # Convert to positive percentage
                'Number_of_Observations': len(df),
                'Number_of_Roll_Days': df['is_roll_day'].sum(),
                'Value_of_1PT': df['value_of_1pt'].iloc[0]
            }

        except Exception as e:
            self.logger.warning(f"Error calculating some metrics: {e}")
            # Fallback calculations
            summary_stats = {
                'Instrument': instrument,
                'Start_Date': df.index[0].strftime('%Y-%m-%d'),
                'End_Date': df.index[-1].strftime('%Y-%m-%d'),
                'Total_Return_Cumulative': (1 + df['rx_pct']).prod() - 1,
                'Total_Return_Annualized_Pct': df['rx_pct'].mean() * 252 * 100,
                'Annualized_Volatility_Pct': df['rx_pct'].std() * np.sqrt(252) * 100,
                'Sharpe_Ratio': (df['rx_pct'].mean() * 252) / (df['rx_pct'].std() * np.sqrt(252)),
                'Max_Drawdown_Pct': abs((df['cumulative_returns'] / df['cumulative_returns'].cummax() - 1).min() * 100),
                'Number_of_Observations': len(df),
                'Number_of_Roll_Days': df['is_roll_day'].sum(),
                'Value_of_1PT': df['value_of_1pt'].iloc[0]
            }

        summary_df = pd.DataFrame([summary_stats])
        summary_filepath = os.path.join(output_dir, f"{instrument}_summary.csv")
        summary_df.to_csv(summary_filepath, index=False)

        self.logger.info(f"Exported {instrument} continuous series to {filepath}")
        self.logger.info(f"Exported {instrument} summary to {summary_filepath}")

        # Print key statistics
        print(f"\n{instrument} Summary Statistics:")
        print(f"Period: {summary_stats['Start_Date']} to {summary_stats['End_Date']}")
        print(f"Total Return (Cumulative): {summary_stats['Total_Return_Cumulative']:.2%}")
        print(f"Annualized Return: {summary_stats['Total_Return_Annualized_Pct']:.2f}%")
        print(f"Annualized Volatility: {summary_stats['Annualized_Volatility_Pct']:.2f}%")
        print(f"Sharpe Ratio: {summary_stats['Sharpe_Ratio']:.2f}")
        print(f"Max Drawdown: {summary_stats['Max_Drawdown_Pct']:.2f}%")
        print(f"Number of Roll Events: {summary_stats['Number_of_Roll_Days']}")

        return df, summary_stats

    def load_all_instruments(self, output_dir: str = None, start_date: str = "2016-01-01",
                             end_date: str = None):
        """Load and export all configured instruments"""
        if output_dir is None:
            output_dir = self.processed_data_dir

        instruments = ['ES', 'TY']

        for instrument in instruments:
            try:
                self.logger.info(f"Processing {instrument}...")
                self.export_continuous_series(instrument, output_dir, start_date, end_date)
            except Exception as e:
                self.logger.error(f"Failed to process {instrument}: {e}")
                import traceback
                traceback.print_exc()


# Example usage
if __name__ == "__main__":
    # Initialize the loader
    loader = ExcelFuturesLoader()

    try:
        # Load S&P 500 E-Mini and US 10Y Note data
        loader.load_all_instruments(
            start_date="2016-01-01"
        )

        print("Data processing completed successfully!")
        print("Check the metadata folder for continuous series CSV files:")
        print("- ES_continuous.csv")
        print("- TY_continuous.csv")
        print("- ES_summary.csv")
        print("- TY_summary.csv")

    except Exception as e:
        print(f"Error during data processing: {e}")
        import traceback

        traceback.print_exc()
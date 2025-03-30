# fetchers.py
# Data fetching utilities for Hong Kong market

import pandas as pd
import numpy as np
import datetime as dt
import os
import yfinance as yf
import time


class HKStockDataFetcher:
    """
    Data fetcher for Hong Kong stocks using Yahoo Finance
    """
    
    def __init__(self):
        """Initialize the HKStockDataFetcher"""
        pass
    
    def fetch_daily_data(self, symbol, start_date, end_date):
        """
        Fetch daily bar data for Hong Kong stocks
        
        Parameters:
        symbol (str): Stock symbol with .HK suffix (e.g., '0700.HK')
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        
        Returns:
        DataFrame: Daily OHLCV data
        """
        try:
            # Add .HK suffix if not already present
            if not symbol.endswith('.HK'):
                symbol = f"{symbol}.HK"
            
            # Fetch data from Yahoo Finance
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            # Handle empty result
            if df.empty:
                print(f"No data returned for {symbol}. Returning mock data.")
                return self._generate_mock_data(symbol, start_date, end_date)
            
            # Reset index to have date as column
            df = df.reset_index()
            
            # Rename columns
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            })
            
            # Set date as index
            df = df.set_index('date')
            
            # Add ticker column
            df['ticker'] = symbol
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            print("Returning mock data instead.")
            return self._generate_mock_data(symbol, start_date, end_date)
    
    def fetch_index_data(self, index_symbol, start_date, end_date):
        """
        Fetch daily index data for Hong Kong market
        
        Parameters:
        index_symbol (str): Index symbol (e.g., '^HSI' for Hang Seng Index)
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        
        Returns:
        DataFrame: Daily index data
        """
        try:
            # Fetch data from Yahoo Finance
            df = yf.download(index_symbol, start=start_date, end=end_date, progress=False)
            
            # Handle empty result
            if df.empty:
                print(f"No data returned for {index_symbol}. Returning mock data.")
                return self._generate_mock_data(index_symbol, start_date, end_date)
            
            # Reset index to have date as column
            df = df.reset_index()
            
            # Rename columns
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            })
            
            # Set date as index
            df = df.set_index('date')
            
            # Add ticker column
            df['ticker'] = index_symbol
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {index_symbol}: {e}")
            print("Returning mock data instead.")
            return self._generate_mock_data(index_symbol, start_date, end_date)
    
    def _generate_mock_data(self, symbol, start_date, end_date):
        """
        Generate mock data for testing
        
        Parameters:
        symbol (str): Stock symbol
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        
        Returns:
        DataFrame: Mock daily OHLCV data
        """
        # Convert string dates to datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Create date range (exclude weekends)
        dates = pd.date_range(start=start, end=end, freq='B')  # Business days
        
        # Set seed for reproducibility
        np.random.seed(hash(symbol) % 10000)
        
        # Generate mock price data
        base_price = 50.0 + hash(symbol) % 50  # Different base price based on symbol
        price_data = np.random.normal(0, 1, len(dates)).cumsum() * 0.5 + base_price
        
        # Create OHLCV data
        data = pd.DataFrame(index=dates)
        data['open'] = price_data * (1 + np.random.normal(0, 0.01, len(dates)))
        data['high'] = data['open'] * (1 + np.random.uniform(0, 0.03, len(dates)))
        data['low'] = data['open'] * (1 - np.random.uniform(0, 0.03, len(dates)))
        data['close'] = price_data
        data['adj_close'] = price_data
        data['volume'] = np.random.uniform(1e6, 5e6, len(dates))
        data['ticker'] = symbol
        
        return data
    
    def fetch_multiple_stocks(self, symbols, start_date, end_date, sleep_time=0.5):
        """
        Fetch data for multiple stocks
        
        Parameters:
        symbols (list): List of stock symbols
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        sleep_time (float): Time to sleep between API calls to avoid rate limits
        
        Returns:
        dict: Dictionary of DataFrames with stock symbols as keys
        """
        result = {}
        
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            result[symbol] = self.fetch_daily_data(symbol, start_date, end_date)
            time.sleep(sleep_time)  # Sleep to avoid API rate limits
        
        return result
    
    def fetch_northbound_flow(self, start_date, end_date):
        """
        Fetch northbound flow data through Stock Connect
        
        Parameters:
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        
        Returns:
        DataFrame: Northbound flow data
        
        Note: This is a mock implementation as actual northbound flow data
        requires specialized data sources like Wind or HKEX API
        """
        # This is a mock implementation
        print("Using mock data for northbound flow (actual data requires specialized sources)")
        
        # Convert string dates to datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Create date range (exclude weekends)
        dates = pd.date_range(start=start, end=end, freq='B')  # Business days
        
        # Generate mock flow data
        np.random.seed(42)
        
        # Base flow with some trend and weekly pattern
        base_flow = np.random.normal(0, 500, len(dates)).cumsum()
        weekly_pattern = np.tile([100, 50, 0, -50, -100], len(dates)//5 + 1)[:len(dates)]
        trend = np.linspace(0, 300, len(dates))
        
        flow = base_flow + weekly_pattern + trend
        
        # Create DataFrame
        data = pd.DataFrame({
            'date': dates,
            'northbound_flow': flow,  # In millions HKD
            'northbound_flow_shanghai': flow * 0.6,  # Shanghai connect flow
            'northbound_flow_shenzhen': flow * 0.4,  # Shenzhen connect flow
        })
        
        # Set date as index
        data = data.set_index('date')
        
        return data


class HKStockDataManager:
    """
    Manager class to handle Hong Kong stock data fetching and storage
    """
    
    def __init__(self, data_dir="data"):
        """
        Initialize the HKStockDataManager
        
        Parameters:
        data_dir (str): Directory to store data
        """
        self.data_dir = data_dir
        self.fetcher = HKStockDataFetcher()
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def get_daily_data(self, symbol, start_date, end_date, use_cache=True):
        """
        Get daily data for a stock, using cache if available
        
        Parameters:
        symbol (str): Stock symbol
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        use_cache (bool): Whether to use cached data if available
        
        Returns:
        DataFrame: Daily OHLCV data
        """
        # Add .HK suffix if not already present
        if not symbol.endswith('.HK'):
            symbol = f"{symbol}.HK"
        
        # Define cache file path
        cache_file = os.path.join(self.data_dir, f"{symbol.replace('.', '_')}_{start_date}_{end_date}.csv")
        
        # Check if cache file exists and use_cache is enabled
        if use_cache and os.path.exists(cache_file):
            try:
                # Load data from cache
                df = pd.read_csv(cache_file, index_col='date', parse_dates=True)
                print(f"Loaded data from cache: {cache_file}")
                return df
            except Exception as e:
                print(f"Error loading cached data: {e}")
                print("Fetching fresh data instead.")
        
        # Fetch fresh data
        df = self.fetcher.fetch_daily_data(symbol, start_date, end_date)
        
        # Save to cache if not empty
        if not df.empty and use_cache:
            try:
                df.to_csv(cache_file)
                print(f"Saved data to cache: {cache_file}")
            except Exception as e:
                print(f"Error saving data to cache: {e}")
        
        return df
    
    def get_index_data(self, index_symbol, start_date, end_date, use_cache=True):
        """
        Get daily data for an index, using cache if available
        
        Parameters:
        index_symbol (str): Index symbol
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        use_cache (bool): Whether to use cached data if available
        
        Returns:
        DataFrame: Daily index data
        """
        # Define cache file path
        cache_file = os.path.join(self.data_dir, f"index_{index_symbol.replace('^', '')}_{start_date}_{end_date}.csv")
        
        # Check if cache file exists and use_cache is enabled
        if use_cache and os.path.exists(cache_file):
            try:
                # Load data from cache
                df = pd.read_csv(cache_file, index_col='date', parse_dates=True)
                print(f"Loaded index data from cache: {cache_file}")
                return df
            except Exception as e:
                print(f"Error loading cached index data: {e}")
                print("Fetching fresh data instead.")
        
        # Fetch fresh data
        df = self.fetcher.fetch_index_data(index_symbol, start_date, end_date)
        
        # Save to cache if not empty
        if not df.empty and use_cache:
            try:
                df.to_csv(cache_file)
                print(f"Saved index data to cache: {cache_file}")
            except Exception as e:
                print(f"Error saving index data to cache: {e}")
        
        return df
    
    def get_northbound_flow(self, start_date, end_date, use_cache=True):
        """
        Get northbound flow data, using cache if available
        
        Parameters:
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        use_cache (bool): Whether to use cached data if available
        
        Returns:
        DataFrame: Northbound flow data
        """
        # Define cache file path
        cache_file = os.path.join(self.data_dir, f"northbound_flow_{start_date}_{end_date}.csv")
        
        # Check if cache file exists and use_cache is enabled
        if use_cache and os.path.exists(cache_file):
            try:
                # Load data from cache
                df = pd.read_csv(cache_file, index_col='date', parse_dates=True)
                print(f"Loaded northbound flow data from cache: {cache_file}")
                return df
            except Exception as e:
                print(f"Error loading cached northbound flow data: {e}")
                print("Fetching fresh data instead.")
        
        # Fetch fresh data
        df = self.fetcher.fetch_northbound_flow(start_date, end_date)
        
        # Save to cache if not empty
        if not df.empty and use_cache:
            try:
                df.to_csv(cache_file)
                print(f"Saved northbound flow data to cache: {cache_file}")
            except Exception as e:
                print(f"Error saving northbound flow data to cache: {e}")
        
        return df
    
    def get_ah_pair_data(self, a_symbol, h_symbol, start_date, end_date, use_cache=True):
        """
        Get data for an A-H share pair, using cache if available
        
        Parameters:
        a_symbol (str): A-share symbol (e.g., '601318.SS')
        h_symbol (str): H-share symbol (e.g., '2318.HK')
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        use_cache (bool): Whether to use cached data if available
        
        Returns:
        DataFrame: Daily data for A-H share pair
        """
        # Define cache file path
        cache_file = os.path.join(
            self.data_dir, 
            f"ah_pair_{a_symbol.replace('.', '_')}_{h_symbol.replace('.', '_')}_{start_date}_{end_date}.csv"
        )
        
        # Check if cache file exists and use_cache is enabled
        if use_cache and os.path.exists(cache_file):
            try:
                # Load data from cache
                df = pd.read_csv(cache_file, index_col='date', parse_dates=True)
                print(f"Loaded A-H pair data from cache: {cache_file}")
                return df
            except Exception as e:
                print(f"Error loading cached A-H pair data: {e}")
                print("Fetching fresh data instead.")
        
        # Fetch fresh data
        try:
            # Fetch A-share data
            a_data = self.fetcher.fetch_daily_data(a_symbol, start_date, end_date)
            
            # Fetch H-share data
            h_data = self.fetcher.fetch_daily_data(h_symbol, start_date, end_date)
            
            # Fetch exchange rate data (CNY/HKD)
            exchange_rate_data = self.fetcher.fetch_daily_data('CNY=X', start_date, end_date)
            
            # Drop unnecessary columns and rename to avoid conflicts
            a_data = a_data.drop(['ticker'], axis=1)
            h_data = h_data.drop(['ticker'], axis=1)
            exchange_rate_data = exchange_rate_data[['close']].rename(columns={'close': 'exchange_rate'})
            
            # Rename A and H columns to avoid conflicts
            a_data = a_data.add_prefix('a_')
            h_data = h_data.add_prefix('h_')
            
            # Merge data on date
            merged_data = pd.concat([a_data, h_data, exchange_rate_data], axis=1)
            
            # Calculate AH premium
            merged_data['ah_premium'] = (merged_data['a_close'] / (merged_data['h_close'] * merged_data['exchange_rate']) - 1) * 100
            
            # Drop rows with NaN values
            merged_data = merged_data.dropna()
            
            # Add ticker columns
            merged_data['a_ticker'] = a_symbol
            merged_data['h_ticker'] = h_symbol
            
            # Save to cache if not empty
            if not merged_data.empty and use_cache:
                try:
                    merged_data.to_csv(cache_file)
                    print(f"Saved A-H pair data to cache: {cache_file}")
                except Exception as e:
                    print(f"Error saving A-H pair data to cache: {e}")
            
            return merged_data
            
        except Exception as e:
            print(f"Error fetching A-H pair data: {e}")
            print("Returning mock data instead.")
            
            # Generate mock data
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
            np.random.seed(42)
            
            # Generate mock price data
            a_price = 10.0 + np.random.normal(0, 0.1, len(dates)).cumsum()
            h_price = 9.0 + np.random.normal(0, 0.1, len(dates)).cumsum()
            exchange_rate = np.ones(len(dates)) * 0.9
            
            # Calculate AH premium
            ah_premium = (a_price / (h_price * exchange_rate) - 1) * 100
            
            # Create DataFrame
            mock_data = pd.DataFrame({
                'a_open': a_price * (1 + np.random.normal(0, 0.01, len(dates))),
                'a_high': a_price * (1 + np.random.uniform(0, 0.03, len(dates))),
                'a_low': a_price * (1 - np.random.uniform(0, 0.03, len(dates))),
                'a_close': a_price,
                'a_volume': np.random.uniform(1e6, 5e6, len(dates)),
                'h_open': h_price * (1 + np.random.normal(0, 0.01, len(dates))),
                'h_high': h_price * (1 + np.random.uniform(0, 0.03, len(dates))),
                'h_low': h_price * (1 - np.random.uniform(0, 0.03, len(dates))),
                'h_close': h_price,
                'h_volume': np.random.uniform(1e6, 5e6, len(dates)),
                'exchange_rate': exchange_rate,
                'ah_premium': ah_premium,
                'a_ticker': a_symbol,
                'h_ticker': h_symbol
            }, index=dates)
            
            # Save to cache if use_cache is enabled
            if use_cache:
                try:
                    mock_data.to_csv(cache_file)
                    print(f"Saved mock A-H pair data to cache: {cache_file}")
                except Exception as e:
                    print(f"Error saving mock A-H pair data to cache: {e}")
            
            return mock_data


# Example usage
if __name__ == "__main__":
    # Create data manager
    data_manager = HKStockDataManager()
    
    # Fetch daily data for a stock
    start_date = '2021-01-01'
    end_date = '2021-12-31'
    df = data_manager.get_daily_data('0700', start_date, end_date)
    
    # Print first few rows
    print(df.head())
    
    # Fetch Hang Seng Index
    hsi = data_manager.get_index_data('^HSI', start_date, end_date)
    print("\nHang Seng Index:")
    print(hsi.head())
    
    # Fetch A-H pair data
    ah_pair = data_manager.get_ah_pair_data('601318.SS', '2318.HK', start_date, end_date)
    print("\nA-H Pair Data:")
    print(ah_pair.head())
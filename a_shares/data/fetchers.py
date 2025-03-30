# fetchers.py
# Data fetching utilities for A-shares market

import pandas as pd
import numpy as np
import datetime as dt
import os

class TushareDataFetcher:
    """
    Data fetcher for A-shares using Tushare API
    
    Note: Requires a Tushare API token. Free registration at tushare.pro
    """
    
    def __init__(self, token=None):
        """
        Initialize the TushareDataFetcher
        
        Parameters:
        token (str): Tushare API token
        """
        self.token = token
        self.pro = None
        
        if token:
            self._initialize_api()
    
    def _initialize_api(self):
        """Initialize Tushare API with token"""
        try:
            import tushare as ts
            ts.set_token(self.token)
            self.pro = ts.pro_api()
            print("Successfully connected to Tushare API")
        except Exception as e:
            print(f"Failed to initialize Tushare API: {e}")
            print("You can use mock data for testing purposes")
    
    def fetch_daily_data(self, symbol, start_date, end_date):
        """
        Fetch daily bar data for A-shares
        
        Parameters:
        symbol (str): Stock symbol with exchange (e.g., '000001.SZ')
        start_date (str): Start date in format 'YYYYMMDD'
        end_date (str): End date in format 'YYYYMMDD'
        
        Returns:
        DataFrame: Daily OHLCV data
        """
        if self.pro is None:
            print("Tushare API not initialized. Returning mock data.")
            return self._generate_mock_data(start_date, end_date)
        
        try:
            # Fetch daily data from Tushare
            df = self.pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)
            
            # Handle empty result
            if df.empty:
                print("No data returned from Tushare. Returning mock data.")
                return self._generate_mock_data(start_date, end_date)
            
            # Process data
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            df = df.rename(columns={
                'trade_date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'vol': 'volume'
            })
            
            # Set date as index
            df = df.set_index('date')
            
            return df
            
        except Exception as e:
            print(f"Error fetching data from Tushare: {e}")
            print("Returning mock data instead.")
            return self._generate_mock_data(start_date, end_date)
    
    def fetch_index_data(self, index_code, start_date, end_date):
        """
        Fetch daily index data for A-shares
        
        Parameters:
        index_code (str): Index code (e.g., '000001.SH' for Shanghai Composite)
        start_date (str): Start date in format 'YYYYMMDD'
        end_date (str): End date in format 'YYYYMMDD'
        
        Returns:
        DataFrame: Daily index data
        """
        if self.pro is None:
            print("Tushare API not initialized. Returning mock data.")
            return self._generate_mock_data(start_date, end_date)
        
        try:
            # Fetch index data from Tushare
            df = self.pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
            
            # Handle empty result
            if df.empty:
                print("No data returned from Tushare. Returning mock data.")
                return self._generate_mock_data(start_date, end_date)
            
            # Process data
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            df = df.rename(columns={
                'trade_date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'vol': 'volume'
            })
            
            # Set date as index
            df = df.set_index('date')
            
            return df
            
        except Exception as e:
            print(f"Error fetching index data from Tushare: {e}")
            print("Returning mock data instead.")
            return self._generate_mock_data(start_date, end_date)
    
    def _generate_mock_data(self, start_date, end_date):
        """
        Generate mock data for testing
        
        Parameters:
        start_date (str): Start date in format 'YYYYMMDD'
        end_date (str): End date in format 'YYYYMMDD'
        
        Returns:
        DataFrame: Mock daily OHLCV data
        """
        # Convert string dates to datetime
        start = dt.datetime.strptime(start_date, '%Y%m%d')
        end = dt.datetime.strptime(end_date, '%Y%m%d')
        
        # Create date range
        dates = pd.date_range(start=start, end=end, freq='B')  # Business days
        
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Generate mock price data
        base_price = 50.0
        price_data = np.random.normal(0, 1, len(dates)).cumsum() * 0.5 + base_price
        
        # Create OHLCV data
        data = pd.DataFrame(index=dates)
        data['open'] = price_data * (1 + np.random.normal(0, 0.01, len(dates)))
        data['high'] = data['open'] * (1 + np.random.uniform(0, 0.03, len(dates)))
        data['low'] = data['open'] * (1 - np.random.uniform(0, 0.03, len(dates)))
        data['close'] = price_data
        data['volume'] = np.random.uniform(1e6, 5e6, len(dates))
        
        # Additional A-shares specific fields
        data['pre_close'] = data['close'].shift(1).fillna(data['open'][0] * 0.99)
        data['change'] = data['close'] - data['pre_close']
        data['pct_chg'] = data['change'] / data['pre_close'] * 100
        data['amount'] = data['volume'] * data['close'] / 1000  # in thousands
        data['turn'] = np.random.uniform(1, 10, len(dates))  # turnover ratio
        
        # Add stock code and date columns
        data['ts_code'] = '000001.SZ'
        
        # Reset index to have date as column
        data = data.reset_index()
        data = data.rename(columns={'index': 'date'})
        
        # Set date as index again
        data = data.set_index('date')
        
        return data

    def fetch_stock_list(self, market=None):
        """
        Fetch list of stocks in A-shares market
        
        Parameters:
        market (str, optional): Market type ('SSE' for Shanghai, 'SZSE' for Shenzhen)
        
        Returns:
        DataFrame: Stock list
        """
        if self.pro is None:
            print("Tushare API not initialized. Returning mock data.")
            return self._generate_mock_stock_list()
        
        try:
            # Fetch stock list from Tushare
            if market:
                df = self.pro.stock_basic(exchange=market, list_status='L')
            else:
                df = self.pro.stock_basic(list_status='L')
            
            # Handle empty result
            if df.empty:
                print("No data returned from Tushare. Returning mock data.")
                return self._generate_mock_stock_list()
            
            return df
            
        except Exception as e:
            print(f"Error fetching stock list from Tushare: {e}")
            print("Returning mock data instead.")
            return self._generate_mock_stock_list()
    
    def _generate_mock_stock_list(self):
        """
        Generate mock stock list for testing
        
        Returns:
        DataFrame: Mock stock list
        """
        # Create mock stock list
        data = {
            'ts_code': ['000001.SZ', '000002.SZ', '000063.SZ', '000333.SZ', '000651.SZ',
                        '600000.SH', '600519.SH', '600036.SH', '601318.SH', '601857.SH'],
            'symbol': ['000001', '000002', '000063', '000333', '000651',
                      '600000', '600519', '600036', '601318', '601857'],
            'name': ['平安银行', '万科A', '中兴通讯', '美的集团', '格力电器',
                     '浦发银行', '贵州茅台', '招商银行', '中国平安', '中国石油'],
            'area': ['深圳', '深圳', '深圳', '佛山', '珠海',
                     '上海', '贵州', '深圳', '深圳', '北京'],
            'industry': ['银行', '房地产', '通信设备', '家电', '家电',
                        '银行', '白酒', '银行', '保险', '石油'],
            'market': ['主板', '主板', '主板', '主板', '主板',
                       '主板', '主板', '主板', '主板', '主板'],
        }
        
        return pd.DataFrame(data)


class ASharesDataManager:
    """
    Manager class to handle A-shares data fetching and storage
    """
    
    def __init__(self, data_dir="data", token=None):
        """
        Initialize the ASharesDataManager
        
        Parameters:
        data_dir (str): Directory to store data
        token (str): Tushare API token
        """
        self.data_dir = data_dir
        self.fetcher = TushareDataFetcher(token)
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def get_daily_data(self, symbol, start_date, end_date, use_cache=True):
        """
        Get daily data for a stock, using cache if available
        
        Parameters:
        symbol (str): Stock symbol
        start_date (str): Start date in format 'YYYYMMDD'
        end_date (str): End date in format 'YYYYMMDD'
        use_cache (bool): Whether to use cached data if available
        
        Returns:
        DataFrame: Daily OHLCV data
        """
        # Define cache file path
        cache_file = os.path.join(self.data_dir, f"{symbol}_{start_date}_{end_date}.csv")
        
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
    
    def get_index_data(self, index_code, start_date, end_date, use_cache=True):
        """
        Get daily data for an index, using cache if available
        
        Parameters:
        index_code (str): Index code
        start_date (str): Start date in format 'YYYYMMDD'
        end_date (str): End date in format 'YYYYMMDD'
        use_cache (bool): Whether to use cached data if available
        
        Returns:
        DataFrame: Daily index data
        """
        # Define cache file path
        cache_file = os.path.join(self.data_dir, f"index_{index_code}_{start_date}_{end_date}.csv")
        
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
        df = self.fetcher.fetch_index_data(index_code, start_date, end_date)
        
        # Save to cache if not empty
        if not df.empty and use_cache:
            try:
                df.to_csv(cache_file)
                print(f"Saved index data to cache: {cache_file}")
            except Exception as e:
                print(f"Error saving index data to cache: {e}")
        
        return df


# Example usage
if __name__ == "__main__":
    # Create data manager
    data_manager = ASharesDataManager()
    
    # Fetch daily data for a stock
    start_date = '20210101'
    end_date = '20211231'
    df = data_manager.get_daily_data('000001.SZ', start_date, end_date)
    
    # Print first few rows
    print(df.head())
# fetchers.py
# Data fetching utilities for US market

import pandas as pd
import numpy as np
import datetime as dt
import os
import yfinance as yf
import time


class USMarketDataFetcher:
    """
    Data fetcher for US market using Yahoo Finance
    """
    
    def __init__(self):
        """Initialize the USMarketDataFetcher"""
        pass
    
    def fetch_daily_data(self, symbol, start_date, end_date):
        """
        Fetch daily bar data for US stocks
        
        Parameters:
        symbol (str): Stock symbol (e.g., 'AAPL')
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        
        Returns:
        DataFrame: Daily OHLCV data
        """
        try:
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
        Fetch daily index data for US market
        
        Parameters:
        index_symbol (str): Index symbol (e.g., '^GSPC' for S&P 500)
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
        base_price = 100.0 + hash(symbol) % 100  # Different base price based on symbol
        
        # Add some randomness to the price trend
        trend = np.random.choice([-0.1, 0, 0.1, 0.2])  # Annual trend
        daily_trend = (1 + trend) ** (1/252) - 1  # Convert to daily
        
        # Generate prices with trend and randomness
        prices = [base_price]
        for i in range(1, len(dates)):
            # Calculate new price with trend and volatility
            daily_return = daily_trend + np.random.normal(0, 0.015)  # 1.5% daily volatility
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)
        
        # Create OHLCV data
        data = pd.DataFrame(index=dates)
        data['open'] = [p * (1 + np.random.normal(0, 0.005)) for p in prices]  # 0.5% variation
        data['high'] = [max(o, c) * (1 + np.random.uniform(0, 0.01)) for o, c in zip(data['open'], prices)]  # 0-1% higher
        data['low'] = [min(o, c) * (1 - np.random.uniform(0, 0.01)) for o, c in zip(data['open'], prices)]  # 0-1% lower
        data['close'] = prices
        data['adj_close'] = prices
        
        # Generate volume with occasional spikes
        base_volume = 1e6 * (1 + hash(symbol) % 10)  # Different base volume based on symbol
        volumes = []
        for i in range(len(dates)):
            # Occasional volume spikes (5% chance)
            if np.random.random() < 0.05:
                vol = base_volume * np.random.uniform(3, 5)
            else:
                vol = base_volume * np.random.uniform(0.5, 1.5)
            volumes.append(vol)
        
        data['volume'] = volumes
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
    
    def fetch_sector_data(self, start_date, end_date):
        """
        Fetch sector ETF data for sector analysis
        
        Parameters:
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        
        Returns:
        DataFrame: Sector ETF data
        """
        # Define sector ETFs
        sector_etfs = {
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLE': 'Energy',
            'XLF': 'Financials',
            'XLV': 'Health Care',
            'XLI': 'Industrials',
            'XLB': 'Materials',
            'XLK': 'Technology',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate'
        }
        
        # Fetch data for all sector ETFs
        sector_data = {}
        for symbol in sector_etfs.keys():
            try:
                df = self.fetch_daily_data(symbol, start_date, end_date)
                df['sector'] = sector_etfs[symbol]
                sector_data[symbol] = df
                time.sleep(0.5)  # Sleep to avoid API rate limits
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        # Check if we got any data
        if not sector_data:
            print("No sector data retrieved. Returning mock data.")
            return self._generate_mock_sector_data(sector_etfs, start_date, end_date)
        
        # Combine all sector data
        combined_data = pd.concat(sector_data.values())
        
        return combined_data
    
    def _generate_mock_sector_data(self, sector_etfs, start_date, end_date):
        """
        Generate mock sector ETF data
        
        Parameters:
        sector_etfs (dict): Dictionary of sector ETF symbols and names
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        
        Returns:
        DataFrame: Mock sector ETF data
        """
        sector_data = {}
        
        for symbol, sector in sector_etfs.items():
            # Generate mock data for this ETF
            df = self._generate_mock_data(symbol, start_date, end_date)
            df['sector'] = sector
            sector_data[symbol] = df
        
        # Combine all sector data
        combined_data = pd.concat(sector_data.values())
        
        return combined_data
    
    def fetch_fundamental_data(self, symbol):
        """
        Fetch fundamental data for a US stock
        
        Parameters:
        symbol (str): Stock symbol
        
        Returns:
        dict: Dictionary with fundamental data
        """
        try:
            # Get ticker info from yfinance
            ticker = yf.Ticker(symbol)
            
            # Get basic info
            info = ticker.info
            
            # Extract relevant fundamental data
            fundamentals = {
                'ticker': symbol,
                'sector': info.get('sector', None),
                'industry': info.get('industry', None),
                'market_cap': info.get('marketCap', None),
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'peg_ratio': info.get('pegRatio', None),
                'price_to_book': info.get('priceToBook', None),
                'dividend_yield': info.get('dividendYield', None) * 100 if info.get('dividendYield', None) else None,
                'profit_margins': info.get('profitMargins', None) * 100 if info.get('profitMargins', None) else None,
                'beta': info.get('beta', None)
            }
            
            return fundamentals
            
        except Exception as e:
            print(f"Error fetching fundamental data for {symbol}: {e}")
            print("Returning mock fundamental data instead.")
            return self._generate_mock_fundamental_data(symbol)
    
    def _generate_mock_fundamental_data(self, symbol):
        """
        Generate mock fundamental data for testing
        
        Parameters:
        symbol (str): Stock symbol
        
        Returns:
        dict: Mock fundamental data
        """
        # Set seed for reproducibility
        np.random.seed(hash(symbol) % 10000)
        
        # List of sectors and industries
        sectors = ['Technology', 'Consumer Cyclical', 'Financial Services', 'Healthcare', 'Industrials', 'Communication Services', 'Energy', 'Basic Materials', 'Consumer Defensive', 'Real Estate', 'Utilities']
        
        industries = {
            'Technology': ['Software', 'Hardware', 'Semiconductors', 'IT Services'],
            'Consumer Cyclical': ['Auto Manufacturers', 'Retail', 'Leisure', 'Restaurants'],
            'Financial Services': ['Banks', 'Insurance', 'Asset Management', 'Credit Services'],
            'Healthcare': ['Biotechnology', 'Medical Devices', 'Pharmaceuticals', 'Healthcare Plans'],
            'Industrials': ['Aerospace & Defense', 'Business Services', 'Transportation', 'Construction'],
            'Communication Services': ['Telecom', 'Media', 'Entertainment', 'Interactive Media'],
            'Energy': ['Oil & Gas E&P', 'Oil & Gas Integrated', 'Oil & Gas Midstream', 'Renewable Energy'],
            'Basic Materials': ['Chemicals', 'Metals & Mining', 'Paper & Forest Products', 'Building Materials'],
            'Consumer Defensive': ['Packaged Foods', 'Beverages', 'Household Products', 'Personal Products'],
            'Real Estate': ['REIT', 'Real Estate Services', 'Real Estate Development', 'Real Estate Operating Companies'],
            'Utilities': ['Utilities—Regulated Electric', 'Utilities—Regulated Gas', 'Utilities—Diversified', 'Utilities—Independent Power Producers']
        }
        
        # Choose a sector based on symbol
        sector = sectors[hash(symbol) % len(sectors)]
        
        # Choose an industry based on sector
        industry = industries[sector][hash(symbol) % len(industries[sector])]
        
        # Generate mock fundamental data
        fundamentals = {
            'ticker': symbol,
            'sector': sector,
            'industry': industry,
            'market_cap': np.random.uniform(1e9, 500e9),  # 1B to 500B market cap
            'pe_ratio': np.random.uniform(10, 40),  # PE ratio between 10 and 40
            'forward_pe': np.random.uniform(8, 35),  # Forward PE between 8 and 35
            'peg_ratio': np.random.uniform(0.5, 3),  # PEG ratio between 0.5 and 3
            'price_to_book': np.random.uniform(1, 10),  # P/B ratio between 1 and 10
            'dividend_yield': np.random.uniform(0, 5),  # Dividend yield between 0% and 5%
            'profit_margins': np.random.uniform(5, 30),  # Profit margins between 5% and 30%
            'beta': np.random.uniform(0.5, 2)  # Beta between 0.5 and 2
        }
        
        return fundamentals
    
    def fetch_economic_data(self, indicator, start_date, end_date):
        """
        Mock function to fetch economic data
        
        Parameters:
        indicator (str): Economic indicator code
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        
        Returns:
        DataFrame: Economic indicator data
        """
        # This is a mock implementation
        print(f"Using mock data for economic indicator: {indicator}")
        
        # Convert string dates to datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Create monthly date range
        dates = pd.date_range(start=start, end=end, freq='M')
        
        # Set seed for reproducibility
        np.random.seed(hash(indicator) % 10000)
        
        # Generate base value based on indicator
        if indicator == 'UNRATE':  # Unemployment rate
            base_value = 5.0
            mock_values = np.random.normal(0, 0.2, len(dates)).cumsum() + base_value
            mock_values = np.clip(mock_values, 3, 10)  # Keep between 3% and 10%
            unit = '%'
        elif indicator == 'FEDFUNDS':  # Federal funds rate
            base_value = 2.0
            mock_values = np.random.normal(0, 0.1, len(dates)).cumsum() + base_value
            mock_values = np.clip(mock_values, 0, 5)  # Keep between 0% and 5%
            unit = '%'
        elif indicator == 'CPI':  # Consumer Price Index
            base_value = 260
            mock_values = np.zeros(len(dates)) + base_value
            for i in range(1, len(dates)):
                # Monthly inflation between 0.1% and 0.5%
                monthly_inflation = np.random.uniform(0.001, 0.005)
                mock_values[i] = mock_values[i-1] * (1 + monthly_inflation)
            unit = 'Index'
        elif indicator == 'GDP':  # GDP (quarterly)
            # Resample to quarterly
            dates = pd.date_range(start=start, end=end, freq='Q')
            base_value = 22000  # Billions of dollars
            mock_values = np.zeros(len(dates)) + base_value
            for i in range(1, len(dates)):
                # Quarterly growth between 0.5% and 1.5%
                quarterly_growth = np.random.uniform(0.005, 0.015)
                mock_values[i] = mock_values[i-1] * (1 + quarterly_growth)
            unit = 'Billions of $'
        else:  # Generic economic indicator
            base_value = 100
            mock_values = np.random.normal(0, 1, len(dates)).cumsum() + base_value
            unit = 'Index'
        
        # Create DataFrame
        data = pd.DataFrame({
            'date': dates,
            'value': mock_values,
            'indicator': indicator,
            'unit': unit
        })
        
        # Set date as index
        data = data.set_index('date')
        
        return data


class USMarketDataManager:
    """
    Manager class to handle US market data fetching and storage
    """
    
    def __init__(self, data_dir="data"):
        """
        Initialize the USMarketDataManager
        
        Parameters:
        data_dir (str): Directory to store data
        """
        self.data_dir = data_dir
        self.fetcher = USMarketDataFetcher()
        
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
    
    def get_sector_data(self, start_date, end_date, use_cache=True):
        """
        Get sector ETF data, using cache if available
        
        Parameters:
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        use_cache (bool): Whether to use cached data if available
        
        Returns:
        DataFrame: Sector ETF data
        """
        # Define cache file path
        cache_file = os.path.join(self.data_dir, f"sector_data_{start_date}_{end_date}.csv")
        
        # Check if cache file exists and use_cache is enabled
        if use_cache and os.path.exists(cache_file):
            try:
                # Load data from cache
                df = pd.read_csv(cache_file, index_col='date', parse_dates=True)
                print(f"Loaded sector data from cache: {cache_file}")
                return df
            except Exception as e:
                print(f"Error loading cached sector data: {e}")
                print("Fetching fresh data instead.")
        
        # Fetch fresh data
        df = self.fetcher.fetch_sector_data(start_date, end_date)
        
        # Save to cache if not empty
        if not df.empty and use_cache:
            try:
                df.to_csv(cache_file)
                print(f"Saved sector data to cache: {cache_file}")
            except Exception as e:
                print(f"Error saving sector data to cache: {e}")
        
        return df
    
    def get_fundamental_data(self, symbol, use_cache=True):
        """
        Get fundamental data for a stock, using cache if available
        
        Parameters:
        symbol (str): Stock symbol
        use_cache (bool): Whether to use cached data if available
        
        Returns:
        dict: Fundamental data
        """
        # Define cache file path
        cache_file = os.path.join(self.data_dir, f"fundamentals_{symbol}.csv")
        
        # Check if cache file exists and use_cache is enabled
        if use_cache and os.path.exists(cache_file):
            try:
                # Load data from cache
                df = pd.read_csv(cache_file)
                fundamentals = df.iloc[0].to_dict()
                print(f"Loaded fundamental data from cache: {cache_file}")
                return fundamentals
            except Exception as e:
                print(f"Error loading cached fundamental data: {e}")
                print("Fetching fresh data instead.")
        
        # Fetch fresh data
        fundamentals = self.fetcher.fetch_fundamental_data(symbol)
        
        # Save to cache if not empty
        if fundamentals and use_cache:
            try:
                pd.DataFrame([fundamentals]).to_csv(cache_file, index=False)
                print(f"Saved fundamental data to cache: {cache_file}")
            except Exception as e:
                print(f"Error saving fundamental data to cache: {e}")
        
        return fundamentals
    
    def get_multiple_stocks(self, symbols, start_date, end_date, use_cache=True):
        """
        Get data for multiple stocks, using cache if available
        
        Parameters:
        symbols (list): List of stock symbols
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        use_cache (bool): Whether to use cached data if available
        
        Returns:
        dict: Dictionary of DataFrames with stock symbols as keys
        """
        result = {}
        
        for symbol in symbols:
            print(f"Getting data for {symbol}...")
            result[symbol] = self.get_daily_data(symbol, start_date, end_date, use_cache)
        
        return result


# Example usage
if __name__ == "__main__":
    # Create data manager
    data_manager = USMarketDataManager()
    
    # Fetch daily data for a stock
    start_date = '2021-01-01'
    end_date = '2021-12-31'
    df = data_manager.get_daily_data('AAPL', start_date, end_date)
    
    # Print first few rows
    print(df.head())
    
    # Fetch S&P 500 index
    sp500 = data_manager.get_index_data('^GSPC', start_date, end_date)
    print("\nS&P 500:")
    print(sp500.head())
    
    # Fetch fundamental data
    fundamentals = data_manager.get_fundamental_data('AAPL')
    print("\nFundamental Data:")
    for key, value in fundamentals.items():
        print(f"{key}: {value}")
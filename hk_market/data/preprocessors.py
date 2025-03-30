# preprocessors.py
# Data preprocessing utilities for Hong Kong market

import pandas as pd
import numpy as np
from scipy import stats
import datetime as dt

class HKStockPreprocessor:
    """
    Preprocessor for Hong Kong stock data
    """
    
    def __init__(self):
        """Initialize the preprocessor"""
        pass
    
    def clean_data(self, df):
        """
        Clean the price data by handling missing values and outliers
        
        Parameters:
        df (DataFrame): Raw price data
        
        Returns:
        DataFrame: Cleaned data
        """
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Handle outliers
        data = self._handle_outliers(data)
        
        # Ensure correct data types
        data = self._ensure_data_types(data)
        
        return data
    
    def _handle_missing_values(self, df):
        """
        Handle missing values in the data
        
        Parameters:
        df (DataFrame): Raw data
        
        Returns:
        DataFrame: Data with missing values handled
        """
        # Make a copy
        data = df.copy()
        
        # Fill missing values in OHLC with previous value
        for col in ['open', 'high', 'low', 'close', 'adj_close']:
            if col in data.columns:
                data[col] = data[col].fillna(method='ffill')
        
        # Fill missing values in volume with median
        if 'volume' in data.columns:
            data['volume'] = data['volume'].fillna(data['volume'].median())
        
        # Drop any remaining rows with missing values in essential columns
        essential_cols = [c for c in ['open', 'high', 'low', 'close'] if c in data.columns]
        data = data.dropna(subset=essential_cols)
        
        return data
    
    def _handle_outliers(self, df, method='zscore', threshold=3.0):
        """
        Handle outliers in the data
        
        Parameters:
        df (DataFrame): Raw data
        method (str): Method to identify outliers ('zscore', 'iqr')
        threshold (float): Threshold for outlier detection
        
        Returns:
        DataFrame: Data with outliers handled
        """
        # Make a copy
        data = df.copy()
        
        # Price columns to check for outliers
        price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in data.columns]
        
        # Apply outlier detection method
        if method == 'zscore':
            # Z-score method
            for col in price_cols:
                zscores = stats.zscore(data[col])
                outliers = abs(zscores) > threshold
                
                # Replace outliers with previous valid value
                data.loc[outliers, col] = np.nan
                data[col] = data[col].fillna(method='ffill')
                
        elif method == 'iqr':
            # IQR method
            for col in price_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
                
                # Replace outliers with previous valid value
                data.loc[outliers, col] = np.nan
                data[col] = data[col].fillna(method='ffill')
        
        return data
    
    def _ensure_data_types(self, df):
        """
        Ensure correct data types for columns
        
        Parameters:
        df (DataFrame): Data
        
        Returns:
        DataFrame: Data with correct types
        """
        # Make a copy
        data = df.copy()
        
        # Ensure price columns are float
        price_cols = [c for c in ['open', 'high', 'low', 'close', 'adj_close'] if c in data.columns]
        for col in price_cols:
            data[col] = data[col].astype(float)
        
        # Ensure volume is float
        if 'volume' in data.columns:
            data['volume'] = data['volume'].astype(float)
        
        return data
    
    def add_technical_indicators(self, df):
        """
        Add common technical indicators to the data
        
        Parameters:
        df (DataFrame): Price data
        
        Returns:
        DataFrame: Data with technical indicators
        """
        # Make a copy
        data = df.copy()
        
        # Check if required columns exist
        if 'close' not in data.columns:
            raise ValueError("DataFrame must have 'close' column")
        
        # Add moving averages
        data = self.add_moving_averages(data)
        
        # Add RSI
        data = self.add_rsi(data)
        
        # Add Bollinger Bands
        data = self.add_bollinger_bands(data)
        
        # Add MACD
        data = self.add_macd(data)
        
        # Add ATR
        if all(col in data.columns for col in ['high', 'low', 'close']):
            data = self.add_atr(data)
        
        # Add extra indicators
        data = self.add_volume_indicators(data)
        
        return data
    
    def add_moving_averages(self, df, periods=[5, 10, 20, 60]):
        """
        Add moving averages to the data
        
        Parameters:
        df (DataFrame): Price data
        periods (list): List of periods for moving averages
        
        Returns:
        DataFrame: Data with moving averages
        """
        # Make a copy
        data = df.copy()
        
        # Add simple moving averages
        for period in periods:
            data[f'ma_{period}'] = data['close'].rolling(window=period).mean()
        
        # Add exponential moving averages
        for period in periods:
            data[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False).mean()
        
        return data
    
    def add_rsi(self, df, periods=[14]):
        """
        Add Relative Strength Index (RSI) to the data
        
        Parameters:
        df (DataFrame): Price data
        periods (list): List of periods for RSI
        
        Returns:
        DataFrame: Data with RSI
        """
        # Make a copy
        data = df.copy()
        
        # Calculate RSI for each period
        for period in periods:
            # Calculate price changes
            delta = data['close'].diff()
            
            # Separate gains and losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Calculate average gain and loss
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            data[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        return data
    
    def add_bollinger_bands(self, df, period=20, num_std=2):
        """
        Add Bollinger Bands to the data
        
        Parameters:
        df (DataFrame): Price data
        period (int): Period for moving average
        num_std (int): Number of standard deviations
        
        Returns:
        DataFrame: Data with Bollinger Bands
        """
        # Make a copy
        data = df.copy()
        
        # Calculate middle band (SMA)
        data['bb_middle'] = data['close'].rolling(window=period).mean()
        
        # Calculate standard deviation
        rolling_std = data['close'].rolling(window=period).std()
        
        # Calculate upper and lower bands
        data['bb_upper'] = data['bb_middle'] + (rolling_std * num_std)
        data['bb_lower'] = data['bb_middle'] - (rolling_std * num_std)
        
        # Calculate %B
        data['bb_pct_b'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Calculate Bandwidth
        data['bb_bandwidth'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        return data
    
    def add_macd(self, df, fast_period=12, slow_period=26, signal_period=9):
        """
        Add Moving Average Convergence Divergence (MACD) to the data
        
        Parameters:
        df (DataFrame): Price data
        fast_period (int): Period for fast EMA
        slow_period (int): Period for slow EMA
        signal_period (int): Period for signal line
        
        Returns:
        DataFrame: Data with MACD
        """
        # Make a copy
        data = df.copy()
        
        # Calculate fast and slow EMAs
        fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        data['macd'] = fast_ema - slow_ema
        
        # Calculate signal line
        data['macd_signal'] = data['macd'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        data['macd_hist'] = data['macd'] - data['macd_signal']
        
        return data
    
    def add_atr(self, df, period=14):
        """
        Add Average True Range (ATR) to the data
        
        Parameters:
        df (DataFrame): Price data with high, low, close columns
        period (int): Period for ATR
        
        Returns:
        DataFrame: Data with ATR
        """
        # Make a copy
        data = df.copy()
        
        # Calculate true range
        data['tr0'] = abs(data['high'] - data['low'])
        data['tr1'] = abs(data['high'] - data['close'].shift())
        data['tr2'] = abs(data['low'] - data['close'].shift())
        data['tr'] = data[['tr0', 'tr1', 'tr2']].max(axis=1)
        
        # Calculate ATR
        data[f'atr_{period}'] = data['tr'].rolling(window=period).mean()
        
        # Drop temporary columns
        data = data.drop(['tr0', 'tr1', 'tr2', 'tr'], axis=1)
        
        return data
    
    def add_volume_indicators(self, df):
        """
        Add volume-based indicators
        
        Parameters:
        df (DataFrame): Price data with volume column
        
        Returns:
        DataFrame: Data with volume indicators
        """
        # Make a copy
        data = df.copy()
        
        # Check if volume column exists
        if 'volume' not in data.columns:
            print("Warning: DataFrame missing 'volume' column for volume indicators")
            return data
        
        # Add volume moving averages
        for period in [5, 10, 20]:
            data[f'volume_ma_{period}'] = data['volume'].rolling(window=period).mean()
        
        # Add volume relative to moving average
        data['volume_ratio'] = data['volume'] / data['volume_ma_20']
        
        # Add volume rate of change
        data['volume_roc'] = data['volume'].pct_change(1) * 100
        
        # Add On-Balance Volume (OBV)
        data['obv'] = 0
        data.loc[1:, 'obv'] = (
            (data['close'] > data['close'].shift(1)).astype(int) * 2 - 1
        ) * data['volume'].loc[1:]
        data['obv'] = data['obv'].cumsum()
        
        # Add Chaikin Money Flow (CMF)
        period = 20
        mf_multiplier = (
            (data['close'] - data['low']) - (data['high'] - data['close'])
        ) / (data['high'] - data['low'])
        mf_volume = mf_multiplier * data['volume']
        data['cmf'] = (
            mf_volume.rolling(window=period).sum() / 
            data['volume'].rolling(window=period).sum()
        )
        
        return data
    
    def add_gap_indicators(self, df):
        """
        Add price gap indicators for Hong Kong stocks
        
        Parameters:
        df (DataFrame): Price data
        
        Returns:
        DataFrame: Data with gap indicators
        """
        # Make a copy
        data = df.copy()
        
        # Check if required columns exist
        if not all(col in data.columns for col in ['open', 'close']):
            print("Warning: DataFrame missing 'open' or 'close' columns for gap indicators")
            return data
        
        # Calculate overnight gap
        data['gap'] = data['open'] / data['close'].shift(1) - 1
        
        # Add gap categories
        data['gap_up'] = (data['gap'] > 0.02).astype(int)  # More than 2% gap up
        data['gap_down'] = (data['gap'] < -0.02).astype(int)  # More than 2% gap down
        
        # Add gap fill indicators
        data['gap_filled'] = ((data['gap'] > 0) & (data['low'] <= data['close'].shift(1))) | \
                             ((data['gap'] < 0) & (data['high'] >= data['close'].shift(1)))
        data['gap_filled'] = data['gap_filled'].astype(int)
        
        return data
    
    def add_ah_premium_indicators(self, df):
        """
        Add A-H premium indicators for dual-listed stocks
        
        Parameters:
        df (DataFrame): Data with A-H premium column
        
        Returns:
        DataFrame: Data with A-H premium indicators
        """
        # Make a copy
        data = df.copy()
        
        # Check if AH premium column exists
        if 'ah_premium' not in data.columns:
            print("Warning: DataFrame missing 'ah_premium' column")
            return data
        
        # Add moving average of AH premium
        for period in [5, 10, 20]:
            data[f'ah_premium_ma_{period}'] = data['ah_premium'].rolling(window=period).mean()
        
        # Add premium z-score (how many standard deviations from historical mean)
        # Use longer history (120 trading days ≈ 6 months) for more stable baseline
        rolling_mean = data['ah_premium'].rolling(window=120).mean()
        rolling_std = data['ah_premium'].rolling(window=120).std()
        data['ah_premium_zscore'] = (data['ah_premium'] - rolling_mean) / rolling_std
        
        # Add premium extreme indicators
        data['ah_premium_extreme_high'] = (data['ah_premium_zscore'] > 2).astype(int)
        data['ah_premium_extreme_low'] = (data['ah_premium_zscore'] < -2).astype(int)
        
        # Add premium change
        data['ah_premium_change'] = data['ah_premium'].diff()
        
        # Add premium trend reversal indicators
        # If premium has been rising for 3 days and then falls, or vice versa
        data['ah_premium_up_streak'] = (data['ah_premium_change'] > 0).astype(int)
        data['ah_premium_down_streak'] = (data['ah_premium_change'] < 0).astype(int)
        
        # Count consecutive days of premium rising/falling
        for i in range(1, len(data)):
            if data['ah_premium_up_streak'].iloc[i] == 1:
                data.iloc[i, data.columns.get_loc('ah_premium_up_streak')] = \
                    data['ah_premium_up_streak'].iloc[i-1] + 1 if data['ah_premium_up_streak'].iloc[i-1] > 0 else 1
            elif data['ah_premium_down_streak'].iloc[i] == 1:
                data.iloc[i, data.columns.get_loc('ah_premium_down_streak')] = \
                    data['ah_premium_down_streak'].iloc[i-1] + 1 if data['ah_premium_down_streak'].iloc[i-1] > 0 else 1
        
        # Add premium reversal indicator
        data['ah_premium_reversal_up'] = ((data['ah_premium_down_streak'].shift(1) >= 3) & 
                                          (data['ah_premium_change'] > 0)).astype(int)
        data['ah_premium_reversal_down'] = ((data['ah_premium_up_streak'].shift(1) >= 3) & 
                                            (data['ah_premium_change'] < 0)).astype(int)
        
        return data
    
    def add_northbound_flow_indicators(self, df, northbound_df):
        """
        Add northbound flow indicators to stock data
        
        Parameters:
        df (DataFrame): Stock price data
        northbound_df (DataFrame): Northbound flow data
        
        Returns:
        DataFrame: Stock data with northbound flow indicators
        """
        # Make a copy
        data = df.copy()
        
        # Ensure northbound_df has a DatetimeIndex
        if not isinstance(northbound_df.index, pd.DatetimeIndex):
            if 'date' in northbound_df.columns:
                northbound_df = northbound_df.set_index('date')
            else:
                print("Warning: northbound_df must have a DatetimeIndex or 'date' column")
                return data
        
        # Check if northbound flow column exists
        if 'northbound_flow' not in northbound_df.columns:
            print("Warning: northbound_df missing 'northbound_flow' column")
            return data
        
        # Ensure data has a DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data = data.set_index('date')
            else:
                print("Warning: df must have a DatetimeIndex or 'date' column")
                return data
        
        # Add northbound flow to stock data
        data = data.join(northbound_df[['northbound_flow']], how='left')
        
        # Fill missing values with 0
        data['northbound_flow'] = data['northbound_flow'].fillna(0)
        
        # Add moving averages of northbound flow
        for period in [5, 10, 20]:
            data[f'northbound_flow_ma_{period}'] = data['northbound_flow'].rolling(window=period).mean()
        
        # Add northbound flow momentum (rate of change)
        data['northbound_flow_roc'] = data['northbound_flow'].pct_change(5)
        
        # Add northbound flow z-score (how many standard deviations from historical mean)
        rolling_mean = data['northbound_flow'].rolling(window=60).mean()
        rolling_std = data['northbound_flow'].rolling(window=60).std()
        data['northbound_flow_zscore'] = (data['northbound_flow'] - rolling_mean) / rolling_std
        
        # Add extreme flow indicators
        data['northbound_flow_extreme_high'] = (data['northbound_flow_zscore'] > 2).astype(int)
        data['northbound_flow_extreme_low'] = (data['northbound_flow_zscore'] < -2).astype(int)
        
        # Add flow direction change indicators
        data['northbound_flow_direction'] = np.sign(data['northbound_flow'])
        data['northbound_flow_direction_change'] = (data['northbound_flow_direction'] != 
                                                    data['northbound_flow_direction'].shift(1)).astype(int)
        
        return data
    
    def normalize_features(self, df, method='zscore'):
        """
        Normalize features for machine learning
        
        Parameters:
        df (DataFrame): Data with features
        method (str): Normalization method ('zscore', 'minmax')
        
        Returns:
        DataFrame: Data with normalized features
        """
        # Make a copy
        data = df.copy()
        
        # Identify numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude certain columns from normalization
        exclude_cols = [col for col in numeric_cols if 'gap_' in col or '_extreme_' in col]
        
        # Filter columns to normalize
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
        
        # Apply normalization
        if method == 'zscore':
            for col in cols_to_normalize:
                mean = data[col].mean()
                std = data[col].std()
                if std != 0:
                    data[f'{col}_norm'] = (data[col] - mean) / std
                else:
                    data[f'{col}_norm'] = 0
        
        elif method == 'minmax':
            for col in cols_to_normalize:
                min_val = data[col].min()
                max_val = data[col].max()
                if max_val > min_val:
                    data[f'{col}_norm'] = (data[col] - min_val) / (max_val - min_val)
                else:
                    data[f'{col}_norm'] = 0.5
        
        return data


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2021-01-01', end='2021-12-31')
    np.random.seed(42)
    
    # Generate price data
    close = np.random.normal(0, 1, len(dates)).cumsum() + 100
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': close * (1 + np.random.normal(0, 0.01, len(dates))),
        'high': close * (1 + np.random.uniform(0, 0.03, len(dates))),
        'low': close * (1 - np.random.uniform(0, 0.03, len(dates))),
        'close': close,
        'volume': np.random.uniform(1e6, 5e6, len(dates)),
    }, index=dates)
    
    # Create preprocessor
    preprocessor = HKStockPreprocessor()
    
    # Clean data
    cleaned_data = preprocessor.clean_data(df)
    
    # Add technical indicators
    data_with_indicators = preprocessor.add_technical_indicators(cleaned_data)
    
    # Add gap indicators
    data_with_gap = preprocessor.add_gap_indicators(data_with_indicators)
    
    # Print first few rows
    print(data_with_gap.head())
    
    # Print column names
    print("\nColumns:", data_with_gap.columns.tolist())
    
    # Create sample A-H premium data
    ah_premium = np.random.normal(0, 5, len(dates)).cumsum() % 20 - 10
    
    # Add to DataFrame
    df['ah_premium'] = ah_premium
    
    # Add A-H premium indicators
    data_with_ah = preprocessor.add_ah_premium_indicators(df)
    
    # Print A-H premium indicators
    print("\nA-H Premium Indicators:")
    ah_cols = [col for col in data_with_ah.columns if 'ah_premium' in col]
    print(data_with_ah[ah_cols].head())
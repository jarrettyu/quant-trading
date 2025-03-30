# preprocessors.py
# Data preprocessing utilities for A-shares market

import pandas as pd
import numpy as np
from scipy import stats


class ASharesPreprocessor:
    """
    Preprocessor for A-shares market data
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
        for col in ['open', 'high', 'low', 'close']:
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
        price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in data.columns]
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
        
        # Add moving averages
        for period in periods:
            data[f'ma_{period}'] = data['close'].rolling(window=period).mean()
        
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
    
    def add_turnover_indicators(self, df):
        """
        Add turnover-based indicators specific to A-shares
        
        Parameters:
        df (DataFrame): Price data with volume and turn columns
        
        Returns:
        DataFrame: Data with turnover indicators
        """
        # Make a copy
        data = df.copy()
        
        # Check if required columns exist
        if 'turn' not in data.columns and 'volume' not in data.columns:
            print("Warning: DataFrame missing 'turn' and 'volume' columns for turnover indicators")
            return data
        
        # Use turnover ratio if available, otherwise calculate from volume
        if 'turn' in data.columns:
            turn = data['turn']
        elif 'volume' in data.columns and 'float_share' in data.columns:
            # Calculate turnover ratio
            turn = data['volume'] / data['float_share'] * 100
            data['turn'] = turn
        else:
            # Can't calculate turnover indicators
            return data
        
        # Add turnover moving averages
        for period in [5, 10, 20]:
            data[f'turn_ma_{period}'] = turn.rolling(window=period).mean()
        
        # Add relative turnover (ratio to MA)
        data['rel_turn'] = turn / data['turn_ma_20']
        
        # Add turnover acceleration (change in turnover)
        data['turn_acc'] = turn.diff() / turn.shift()
        
        return data
    
    def add_limit_indicators(self, df):
        """
        Add price limit indicators specific to A-shares
        
        Parameters:
        df (DataFrame): Price data
        
        Returns:
        DataFrame: Data with price limit indicators
        """
        # Make a copy
        data = df.copy()
        
        # Check if required columns exist
        if 'pre_close' not in data.columns:
            if 'close' in data.columns:
                data['pre_close'] = data['close'].shift(1)
            else:
                print("Warning: DataFrame missing 'close' column for limit indicators")
                return data
        
        # Calculate price limits (±10% for most stocks, ±5% for ST stocks)
        # For simplicity, assume all are regular stocks (±10%)
        data['limit_up'] = data['pre_close'] * 1.1
        data['limit_down'] = data['pre_close'] * 0.9
        
        # Add indicators for prices hitting limits
        data['is_limit_up'] = (data['close'] >= data['limit_up'] * 0.995)
        data['is_limit_down'] = (data['close'] <= data['limit_down'] * 1.005)
        
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
        exclude_cols = ['is_limit_up', 'is_limit_down']  # Boolean indicators
        
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
        'turn': np.random.uniform(1, 5, len(dates)),
    }, index=dates)
    
    # Create preprocessor
    preprocessor = ASharesPreprocessor()
    
    # Clean data
    cleaned_data = preprocessor.clean_data(df)
    
    # Add technical indicators
    data_with_indicators = preprocessor.add_technical_indicators(cleaned_data)
    
    # Add A-shares specific indicators
    data_with_ashares_indicators = preprocessor.add_limit_indicators(
        preprocessor.add_turnover_indicators(data_with_indicators)
    )
    
    # Print first few rows
    print(data_with_ashares_indicators.head())
    
    # Print column names
    print("\nColumns:", data_with_ashares_indicators.columns.tolist())
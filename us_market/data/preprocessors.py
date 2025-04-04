# preprocessors.py
# Data preprocessing utilities for US market

import pandas as pd
import numpy as np
from scipy import stats
import datetime as dt

class USMarketPreprocessor:
    """
    Preprocessor for US market data
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
        
        # Add volume indicators
        data = self.add_volume_indicators(data)
        
        # Add momentum indicators
        data = self.add_momentum_indicators(data)
        
        return data
    
    def add_moving_averages(self, df, periods=[5, 10, 20, 50, 200]):
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
        
        # Add golden/death cross signals
        if 50 in periods and 200 in periods:
            # Golden cross: 50-day MA crosses above 200-day MA
            data['golden_cross'] = (data['ma_50'] > data['ma_200']) & (data['ma_50'].shift(1) <= data['ma_200'].shift(1))
            
            # Death cross: 50-day MA crosses below 200-day MA
            data['death_cross'] = (data['ma_50'] < data['ma_200']) & (data['ma_50'].shift(1) >= data['ma_200'].shift(1))
        
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
            
            # Add RSI signal columns
            data[f'rsi_{period}_oversold'] = data[f'rsi_{period}'] < 30
            data[f'rsi_{period}_overbought'] = data[f'rsi_{period}'] > 70
        
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
        
        # Add Bollinger Band signals
        data['bb_upper_touch'] = (data['high'] >= data['bb_upper']) & (data['high'].shift(1) < data['bb_upper'].shift(1))
        data['bb_lower_touch'] = (data['low'] <= data['bb_lower']) & (data['low'].shift(1) > data['bb_lower'].shift(1))
        
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
        
        # Add MACD signals
        data['macd_cross_above'] = (data['macd'] > data['macd_signal']) & (data['macd'].shift(1) <= data['macd_signal'].shift(1))
        data['macd_cross_below'] = (data['macd'] < data['macd_signal']) & (data['macd'].shift(1) >= data['macd_signal'].shift(1))
        
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
        
        # Calculate ATR percentage (ATR/Close)
        data[f'atr_pct_{period}'] = data[f'atr_{period}'] / data['close'] * 100
        
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
        for period in [5, 10, 20, 50]:
            data[f'volume_ma_{period}'] = data['volume'].rolling(window=period).mean()
        
        # Add volume relative to moving average
        data['volume_ratio_20'] = data['volume'] / data['volume_ma_20']
        
        # Add volume rate of change
        data['volume_roc'] = data['volume'].pct_change(1) * 100
        
        # Add On-Balance Volume (OBV)
        data['obv'] = 0
        data.loc[1:, 'obv'] = np.where(
            data['close'].iloc[1:].values > data['close'].iloc[:-1].values,
            data['volume'].iloc[1:].values,
            np.where(
                data['close'].iloc[1:].values < data['close'].iloc[:-1].values,
                -data['volume'].iloc[1:].values,
                0
            )
        )
        data['obv'] = data['obv'].cumsum()
        
        # Add Chaikin Money Flow (CMF)
        period = 20
        mf_multiplier = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        mf_volume = mf_multiplier * data['volume']
        data['cmf'] = mf_volume.rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
        
        # Add volume spike indicator
        data['volume_spike'] = data['volume'] > data['volume_ma_20'] * 2
        
        return data
    
    def add_momentum_indicators(self, df):
        """
        Add momentum-based indicators
        
        Parameters:
        df (DataFrame): Price data
        
        Returns:
        DataFrame: Data with momentum indicators
        """
        # Make a copy
        data = df.copy()
        
        # Add price momentum for different periods
        for period in [1, 5, 10, 20, 60, 120, 252]:
            data[f'momentum_{period}d'] = data['close'].pct_change(period) * 100
        
        # Add Rate of Change (ROC)
        for period in [10, 20, 60]:
            data[f'roc_{period}'] = (data['close'] / data['close'].shift(period) - 1) * 100
        
        # Add Stochastic Oscillator
        period = 14
        data['stoch_k'] = 100 * (data['close'] - data['low'].rolling(window=period).min()) / (data['high'].rolling(window=period).max() - data['low'].rolling(window=period).min())
        data['stoch_d'] = data['stoch_k'].rolling(window=3).mean()
        
        # Add stochastic signals
        data['stoch_oversold'] = data['stoch_k'] < 20
        data['stoch_overbought'] = data['stoch_k'] > 80
        data['stoch_cross_above'] = (data['stoch_k'] > data['stoch_d']) & (data['stoch_k'].shift(1) <= data['stoch_d'].shift(1))
        data['stoch_cross_below'] = (data['stoch_k'] < data['stoch_d']) & (data['stoch_k'].shift(1) >= data['stoch_d'].shift(1))
        
        return data
    
    def add_trend_indicators(self, df):
        """
        Add trend identification indicators
        
        Parameters:
        df (DataFrame): Price data
        
        Returns:
        DataFrame: Data with trend indicators
        """
        # Make a copy
        data = df.copy()
        
        # Add ADX (Average Directional Index)
        period = 14
        
        # Calculate +DM and -DM
        data['up_move'] = data['high'] - data['high'].shift(1)
        data['down_move'] = data['low'].shift(1) - data['low']
        
        data['pos_dm'] = np.where((data['up_move'] > data['down_move']) & (data['up_move'] > 0), data['up_move'], 0)
        data['neg_dm'] = np.where((data['down_move'] > data['up_move']) & (data['down_move'] > 0), data['down_move'], 0)
        
        # Calculate ATR components
        data['tr'] = np.maximum(
            np.maximum(
                data['high'] - data['low'],
                abs(data['high'] - data['close'].shift(1))
            ),
            abs(data['low'] - data['close'].shift(1))
        )
        
        # Calculate smoothed TR, +DM, and -DM
        data['smoothed_tr'] = data['tr'].rolling(window=period).sum()
        data['smoothed_pos_dm'] = data['pos_dm'].rolling(window=period).sum()
        data['smoothed_neg_dm'] = data['neg_dm'].rolling(window=period).sum()
        
        # Calculate +DI and -DI
        data['pos_di'] = 100 * data['smoothed_pos_dm'] / data['smoothed_tr']
        data['neg_di'] = 100 * data['smoothed_neg_dm'] / data['smoothed_tr']
        
        # Calculate directional movement index (DX)
        data['dx'] = 100 * abs(data['pos_di'] - data['neg_di']) / (data['pos_di'] + data['neg_di'])
        
        # Calculate ADX
        data['adx'] = data['dx'].rolling(window=period).mean()
        
        # Add ADX trend signals
        data['strong_trend'] = data['adx'] > 25
        data['weak_trend'] = data['adx'] < 20
        
        # Clean up temporary columns
        data = data.drop(['up_move', 'down_move', 'pos_dm', 'neg_dm', 'tr', 'smoothed_tr', 
                          'smoothed_pos_dm', 'smoothed_neg_dm', 'dx'], axis=1)
        
        # Add Aroon Oscillator
        period = 25
        data['aroon_up'] = 100 * (period - data['high'].rolling(period).apply(lambda x: x.argmax())) / period
        data['aroon_down'] = 100 * (period - data['low'].rolling(period).apply(lambda x: x.argmin())) / period
        data['aroon_osc'] = data['aroon_up'] - data['aroon_down']
        
        # Add trend signals based on Aroon
        data['aroon_bullish'] = data['aroon_osc'] > 50
        data['aroon_bearish'] = data['aroon_osc'] < -50
        
        return data
    
    def add_volatility_indicators(self, df):
        """
        Add volatility indicators
        
        Parameters:
        df (DataFrame): Price data
        
        Returns:
        DataFrame: Data with volatility indicators
        """
        # Make a copy
        data = df.copy()
        
        # Calculate historical volatility
        for period in [10, 20, 60]:
            # Daily returns
            returns = data['close'].pct_change()
            
            # Calculate standard deviation of returns
            data[f'volatility_{period}d'] = returns.rolling(window=period).std() * np.sqrt(252) * 100  # Annualized
        
        # Add volatility ratio (current volatility / longer-term volatility)
        data['volatility_ratio'] = data['volatility_20d'] / data['volatility_60d']
        
        # Add volatility breakout signals
        data['volatility_breakout'] = data['volatility_10d'] > data['volatility_60d'] * 1.5  # 50% higher than baseline
        
        return data
    
    def add_sector_relative_strength(self, df, sector_df):
        """
        Add sector relative strength indicators
        
        Parameters:
        df (DataFrame): Stock price data
        sector_df (DataFrame): Sector ETF price data for stock's sector
        
        Returns:
        DataFrame: Data with sector relative strength indicators
        """
        # Make a copy
        data = df.copy()
        
        # Check inputs
        if 'close' not in data.columns or 'close' not in sector_df.columns:
            print("Warning: DataFrames must have 'close' columns")
            return data
        
        # Calculate stock and sector returns
        stock_returns = data['close'].pct_change()
        sector_returns = sector_df['close'].pct_change()
        
        # Calculate relative strength (stock return - sector return)
        data['sector_rs'] = stock_returns - sector_returns
        
        # Calculate cumulative relative strength
        data['sector_rs_cumulative'] = (1 + data['sector_rs']).cumprod()
        
        # Add relative strength moving averages
        for period in [5, 20, 60]:
            data[f'sector_rs_{period}d'] = data['sector_rs'].rolling(window=period).mean()
        
        # Add relative strength trend indicators
        data['sector_rs_improving'] = data['sector_rs_5d'] > data['sector_rs_20d']
        data['sector_rs_deteriorating'] = data['sector_rs_5d'] < data['sector_rs_20d']
        
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
        
        # Exclude boolean indicators and price/volume columns from normalization
        exclude_patterns = ['_cross', '_oversold', '_overbought', '_spike', '_bullish', '_bearish', 
                           'open', 'high', 'low', 'close', 'volume', 'adj_close']
        
        # Filter columns to normalize
        cols_to_normalize = [col for col in numeric_cols if not any(pattern in col for pattern in exclude_patterns)]
        
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
    preprocessor = USMarketPreprocessor()
    
    # Clean data
    cleaned_data = preprocessor.clean_data(df)
    
    # Add technical indicators
    data_with_indicators = preprocessor.add_technical_indicators(cleaned_data)
    
    # Print first few rows
    print(data_with_indicators.head())
    
    # Print column names
    print("\nColumns:", data_with_indicators.columns.tolist())
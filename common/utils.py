# utils.py
# Common utility functions for quantitative trading strategies

import pandas as pd
import numpy as np
import datetime as dt
import os
import json
import pickle
import warnings
from pathlib import Path


def ensure_datetime_index(df, date_column=None):
    """
    Ensure DataFrame has a DatetimeIndex
    
    Parameters:
    df (DataFrame): Input DataFrame
    date_column (str, optional): Name of column to use as index
    
    Returns:
    DataFrame: DataFrame with DatetimeIndex
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # If DataFrame already has DatetimeIndex, return it
    if isinstance(result.index, pd.DatetimeIndex):
        return result
    
    # If date column is provided, set it as index
    if date_column is not None:
        if date_column in result.columns:
            result[date_column] = pd.to_datetime(result[date_column])
            result = result.set_index(date_column)
            return result
        else:
            warnings.warn(f"Column '{date_column}' not found in DataFrame. Index not changed.")
    
    # Try common date column names
    common_date_cols = ['date', 'Date', 'datetime', 'Datetime', 'timestamp', 'Timestamp']
    for col in common_date_cols:
        if col in result.columns:
            result[col] = pd.to_datetime(result[col])
            result = result.set_index(col)
            return result
    
    # If no date column found, return original DataFrame
    warnings.warn("No date column found. Index not changed.")
    return result


def convert_price_format(df, format_type='ohlc'):
    """
    Convert price DataFrame to a standard format
    
    Parameters:
    df (DataFrame): Input price DataFrame
    format_type (str): Target format type ('ohlc', 'close_only', 'returns')
    
    Returns:
    DataFrame: DataFrame in the specified format
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Ensure DataFrame has DatetimeIndex
    result = ensure_datetime_index(result)
    
    # Map column names to standard format
    column_maps = {
        'o': ['open', 'Open', 'OPEN', 'o', 'O'],
        'h': ['high', 'High', 'HIGH', 'h', 'H'],
        'l': ['low', 'Low', 'LOW', 'l', 'L'],
        'c': ['close', 'Close', 'CLOSE', 'c', 'C', 'adj_close', 'Adj Close'],
        'v': ['volume', 'Volume', 'VOLUME', 'v', 'V'],
    }
    
    # Create standard column mapping
    col_mapping = {}
    for std_col, possible_cols in column_maps.items():
        for col in possible_cols:
            if col in result.columns:
                col_mapping[col] = std_col
                break
    
    # Rename columns
    if col_mapping:
        result = result.rename(columns=col_mapping)
    
    # Format based on requested type
    if format_type == 'ohlc':
        # Check if we have OHLC data
        if not all(col in result.columns for col in ['o', 'h', 'l', 'c']):
            # If we only have close price, replicate it for OHLC
            if 'c' in result.columns:
                result['o'] = result['h'] = result['l'] = result['c']
            else:
                raise ValueError("Cannot convert to OHLC format: insufficient price data")
        
        # Rename to standard OHLC format
        result = result.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close'})
        
        # Ensure float data type
        for col in ['open', 'high', 'low', 'close']:
            result[col] = result[col].astype(float)
        
        if 'v' in result.columns:
            result = result.rename(columns={'v': 'volume'})
            result['volume'] = result['volume'].astype(float)
    
    elif format_type == 'close_only':
        # Check if we have close price
        if 'c' in result.columns:
            result = result[['c']].rename(columns={'c': 'close'})
        elif 'close' in result.columns:
            result = result[['close']]
        else:
            raise ValueError("Cannot convert to close_only format: close price not found")
        
        # Ensure float data type
        result['close'] = result['close'].astype(float)
    
    elif format_type == 'returns':
        # Check if we have close price
        if 'c' in result.columns:
            close_prices = result['c']
        elif 'close' in result.columns:
            close_prices = result['close']
        else:
            raise ValueError("Cannot convert to returns format: close price not found")
        
        # Calculate returns
        result = pd.DataFrame({'returns': close_prices.pct_change()})
        
        # Drop first row with NaN
        result = result.dropna()
    
    else:
        raise ValueError(f"Unknown format_type: {format_type}")
    
    return result


def date_range_to_periods(start_date, end_date, period='1d'):
    """
    Convert date range to list of periods
    
    Parameters:
    start_date (str or datetime): Start date
    end_date (str or datetime): End date
    period (str): Period frequency ('1d', '1w', '1m', '1y')
    
    Returns:
    DataFrame: DataFrame with date ranges for each period
    """
    # Convert string dates to datetime
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Map period to pandas frequency
    freq_map = {
        '1d': 'D',
        '1w': 'W',
        '1m': 'MS',  # Month start
        '1y': 'YS',  # Year start
    }
    
    if period not in freq_map:
        raise ValueError(f"Unknown period: {period}. Use one of: {list(freq_map.keys())}")
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq=freq_map[period])
    
    # Create periods DataFrame
    periods = pd.DataFrame({
        'start_date': dates,
        'end_date': pd.Series(dates).shift(-1) - pd.Timedelta(days=1)
    }).dropna()
    
    # Fix end date for last period
    if len(periods) > 0:
        periods.loc[periods.index[-1], 'end_date'] = end_date
    
    return periods


def calculate_trade_metrics(trades_df):
    """
    Calculate trade performance metrics
    
    Parameters:
    trades_df (DataFrame): DataFrame with trade details
                          (columns: entry_price, exit_price, quantity, pnl)
    
    Returns:
    dict: Dictionary with trade metrics
    """
    # Check if we have any trades
    if len(trades_df) == 0:
        return {
            'num_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'avg_return': 0,
            'avg_bars_held': 0,
        }
    
    # Calculate metrics
    num_trades = len(trades_df)
    win_trades = trades_df[trades_df['pnl'] > 0]
    loss_trades = trades_df[trades_df['pnl'] <= 0]
    
    win_rate = len(win_trades) / num_trades if num_trades > 0 else 0
    
    avg_profit = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
    avg_loss = loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0
    
    total_profit = win_trades['pnl'].sum() if len(win_trades) > 0 else 0
    total_loss = abs(loss_trades['pnl'].sum()) if len(loss_trades) > 0 else 0
    
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Calculate return per trade
    trades_df['return'] = trades_df['pnl'] / (trades_df['entry_price'] * trades_df['quantity'])
    avg_return = trades_df['return'].mean()
    
    # Calculate bars held if we have entry/exit dates
    avg_bars_held = 0
    if 'entry_date' in trades_df.columns and 'exit_date' in trades_df.columns:
        trades_df['bars_held'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
        avg_bars_held = trades_df['bars_held'].mean()
    
    return {
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'avg_return': avg_return,
        'avg_bars_held': avg_bars_held,
    }


def calculate_cagr(start_value, end_value, years):
    """
    Calculate Compound Annual Growth Rate (CAGR)
    
    Parameters:
    start_value (float): Starting value
    end_value (float): Ending value
    years (float): Number of years
    
    Returns:
    float: CAGR as decimal
    """
    if start_value <= 0 or years <= 0:
        return 0
    
    return (end_value / start_value) ** (1 / years) - 1


def save_backtest_results(results, filename, format='json'):
    """
    Save backtest results to file
    
    Parameters:
    results (dict): Dictionary with backtest results
    filename (str): Filename to save to
    format (str): File format ('json', 'pickle', 'csv')
    
    Returns:
    bool: True if successful, False otherwise
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    try:
        # Save based on format
        if format == 'json':
            # Convert pandas objects to json-serializable format
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    if isinstance(value.index, pd.DatetimeIndex):
                        serializable_results[key] = value.reset_index().to_dict('records')
                    else:
                        serializable_results[key] = value.to_dict('records')
                else:
                    serializable_results[key] = value
            
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=4)
        
        elif format == 'pickle':
            with open(filename, 'wb') as f:
                pickle.dump(results, f)
        
        elif format == 'csv':
            # Save DataFrames to CSV
            for key, value in results.items():
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    csv_filename = f"{os.path.splitext(filename)[0]}_{key}.csv"
                    value.to_csv(csv_filename)
                    print(f"Saved {key} to {csv_filename}")
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        return True
    
    except Exception as e:
        print(f"Error saving results: {e}")
        return False


def load_backtest_results(filename, format='json'):
    """
    Load backtest results from file
    
    Parameters:
    filename (str): Filename to load from
    format (str): File format ('json', 'pickle', 'csv')
    
    Returns:
    dict: Dictionary with backtest results
    """
    try:
        # Load based on format
        if format == 'json':
            with open(filename, 'r') as f:
                return json.load(f)
        
        elif format == 'pickle':
            with open(filename, 'rb') as f:
                return pickle.load(f)
        
        elif format == 'csv':
            # Load all CSV files in directory with matching prefix
            directory = os.path.dirname(filename)
            prefix = os.path.splitext(os.path.basename(filename))[0]
            
            results = {}
            for file in os.listdir(directory):
                if file.startswith(prefix) and file.endswith('.csv'):
                    key = file[len(prefix)+1:-4]  # Extract key from filename
                    results[key] = pd.read_csv(os.path.join(directory, file), index_col=0)
            
            return results
        
        else:
            raise ValueError(f"Unknown format: {format}")
    
    except Exception as e:
        print(f"Error loading results: {e}")
        return None


def combine_signals(signals_dict, method='majority'):
    """
    Combine multiple trading signals into a single signal
    
    Parameters:
    signals_dict (dict): Dictionary with signal Series (1=buy, -1=sell, 0=hold)
    method (str): Combination method ('majority', 'unanimous', 'average')
    
    Returns:
    Series: Combined signal
    """
    # Convert to DataFrame
    signals_df = pd.DataFrame(signals_dict)
    
    # Combine based on method
    if method == 'majority':
        # Count number of each signal
        buy_count = (signals_df == 1).sum(axis=1)
        sell_count = (signals_df == -1).sum(axis=1)
        hold_count = (signals_df == 0).sum(axis=1)
        
        # Generate combined signal
        combined = pd.Series(0, index=signals_df.index)
        combined[buy_count > sell_count] = 1
        combined[sell_count > buy_count] = -1
        
        # If equal buy and sell signals, use hold
        return combined
    
    elif method == 'unanimous':
        # Only generate buy/sell signal if all agree
        buy_unanimous = (signals_df == 1).all(axis=1)
        sell_unanimous = (signals_df == -1).all(axis=1)
        
        combined = pd.Series(0, index=signals_df.index)
        combined[buy_unanimous] = 1
        combined[sell_unanimous] = -1
        
        return combined
    
    elif method == 'average':
        # Take average of signals
        avg_signal = signals_df.mean(axis=1)
        
        combined = pd.Series(0, index=signals_df.index)
        combined[avg_signal > 0.3] = 1
        combined[avg_signal < -0.3] = -1
        
        return combined
    
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_position_sizes(signals, prices, capital, position_sizing='equal'):
    """
    Calculate position sizes based on signals and position sizing method
    
    Parameters:
    signals (Series): Series with trading signals (1=buy, -1=sell, 0=hold)
    prices (Series): Series with asset prices
    capital (float): Available capital
    position_sizing (str): Position sizing method ('equal', 'percent_risk', 'volatility')
    
    Returns:
    Series: Position sizes in number of shares/contracts
    """
    # Initialize position sizes
    position_sizes = pd.Series(0, index=signals.index)
    
    # Equal position sizing (equal dollar amount per trade)
    if position_sizing == 'equal':
        for date, signal in signals.items():
            if signal != 0 and date in prices.index:
                price = prices.loc[date]
                shares = int(capital / price)
                position_sizes.loc[date] = shares if signal == 1 else -shares
    
    # Percent risk position sizing (risk a fixed percentage of capital per trade)
    elif position_sizing == 'percent_risk':
        risk_percent = 0.01  # Risk 1% of capital per trade
        for date, signal in signals.items():
            if signal != 0 and date in prices.index:
                price = prices.loc[date]
                risk_amount = capital * risk_percent
                # Assume 5% stop loss for simplicity
                stop_distance = price * 0.05
                shares = int(risk_amount / stop_distance)
                position_sizes.loc[date] = shares if signal == 1 else -shares
    
    # Volatility-based position sizing
    elif position_sizing == 'volatility':
        # Calculate 20-day volatility
        volatility = prices.pct_change().rolling(20).std()
        
        for date, signal in signals.items():
            if signal != 0 and date in prices.index and date in volatility.index:
                price = prices.loc[date]
                vol = volatility.loc[date]
                if vol > 0:
                    # Target 2% max daily loss
                    target_risk = capital * 0.02
                    shares = int(target_risk / (price * vol))
                    position_sizes.loc[date] = shares if signal == 1 else -shares
    
    else:
        raise ValueError(f"Unknown position sizing method: {position_sizing}")
    
    return position_sizes


def resample_data(df, timeframe='1d'):
    """
    Resample OHLCV data to a different timeframe
    
    Parameters:
    df (DataFrame): OHLCV DataFrame
    timeframe (str): Target timeframe ('1d', '1w', '1m', '1h', etc.)
    
    Returns:
    DataFrame: Resampled DataFrame
    """
    # Ensure DataFrame has DatetimeIndex
    df = ensure_datetime_index(df)
    
    # Map timeframe to pandas frequency
    freq_map = {
        '1m': '1min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '1h': '1H',
        '4h': '4H',
        '1d': '1D',
        '1w': '1W',
        '1mo': '1M',
    }
    
    if timeframe not in freq_map:
        raise ValueError(f"Unknown timeframe: {timeframe}. Use one of: {list(freq_map.keys())}")
    
    # Define resampling rules
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Apply resampling
    resampled = df.resample(freq_map[timeframe]).agg(
        {col: ohlc_dict[col] for col in df.columns if col in ohlc_dict}
    )
    
    return resampled


def estimate_transaction_costs(trades_df, market='us', commission_rate=None):
    """
    Estimate transaction costs for a set of trades
    
    Parameters:
    trades_df (DataFrame): DataFrame with trade details (quantity, price)
    market (str): Market type ('us', 'hk', 'china')
    commission_rate (float, optional): Custom commission rate
    
    Returns:
    Series: Transaction costs for each trade
    """
    # Default commission rates by market
    default_rates = {
        'us': 0.0005,  # 0.05%
        'hk': 0.0007,  # 0.07%
        'china': 0.0005,  # 0.05%
    }
    
    # Use custom rate if provided
    rate = commission_rate if commission_rate is not None else default_rates.get(market.lower(), 0.001)
    
    # Calculate trade value
    trade_value = trades_df['quantity'].abs() * trades_df['price']
    
    # Calculate commission
    commission = trade_value * rate
    
    # Add market-specific fees
    if market.lower() == 'hk':
        # Hong Kong fees
        stamp_duty = trade_value * 0.001  # 0.1% stamp duty
        trading_fee = trade_value * 0.00005  # 0.005% trading fee
        transaction_levy = trade_value * 0.000027  # 0.0027% SFC levy
        
        total_cost = commission + stamp_duty + trading_fee + transaction_levy
    
    elif market.lower() == 'china':
        # China A-shares fees
        stamp_duty = trade_value * 0.001  # 0.1% stamp duty (sell only)
        transfer_fee = trade_value * 0.00002  # 0.002% transfer fee
        
        # Apply stamp duty only to sell trades
        is_sell = trades_df.get('side', 'buy') == 'sell'
        stamp_cost = stamp_duty.where(is_sell, 0)
        
        total_cost = commission + stamp_cost + transfer_fee
    
    else:
        # US market or default
        sec_fee = trade_value * 0.0000229  # SEC fee
        
        total_cost = commission + sec_fee
    
    return total_cost


def calculate_strategy_exposure(positions, prices):
    """
    Calculate strategy exposure over time
    
    Parameters:
    positions (DataFrame): DataFrame with position sizes for each asset
    prices (DataFrame): DataFrame with asset prices
    
    Returns:
    DataFrame: DataFrame with strategy exposure metrics
    """
    # Calculate position values
    position_values = positions.multiply(prices)
    
    # Calculate total portfolio value (sum across all positions)
    portfolio_value = position_values.sum(axis=1)
    
    # Calculate long and short exposure
    long_exposure = position_values[position_values > 0].sum(axis=1).fillna(0)
    short_exposure = position_values[position_values < 0].sum(axis=1).fillna(0)
    
    # Calculate net and gross exposure
    net_exposure = long_exposure + short_exposure  # Short exposure is negative
    gross_exposure = long_exposure - short_exposure  # Convert short to positive
    
    # Calculate exposure metrics
    exposure_metrics = pd.DataFrame({
        'portfolio_value': portfolio_value,
        'long_exposure': long_exposure,
        'short_exposure': short_exposure,
        'net_exposure': net_exposure,
        'gross_exposure': gross_exposure,
        'long_pct': long_exposure / portfolio_value,
        'short_pct': -short_exposure / portfolio_value,  # Convert to positive percentage
        'net_pct': net_exposure / portfolio_value,
        'gross_pct': gross_exposure / portfolio_value,
    })
    
    return exposure_metrics


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2021-12-31')
    prices = pd.DataFrame({
        'Close': np.random.normal(100, 5, len(dates)) + np.arange(len(dates)) * 0.05,
        'Open': np.random.normal(100, 5, len(dates)) + np.arange(len(dates)) * 0.05,
        'High': np.random.normal(102, 5, len(dates)) + np.arange(len(dates)) * 0.05,
        'Low': np.random.normal(98, 5, len(dates)) + np.arange(len(dates)) * 0.05,
        'Volume': np.random.normal(1000000, 200000, len(dates)),
        'Date': dates
    })
    
    # Test convert_price_format
    ohlc_df = convert_price_format(prices, 'ohlc')
    print(ohlc_df.head())
    
    # Test date_range_to_periods
    periods = date_range_to_periods('2020-01-01', '2020-12-31', '1m')
    print(periods.head())
    
    # Test calculate_cagr
    cagr = calculate_cagr(100000, 150000, 3)
    print(f"CAGR: {cagr:.2%}")
# double_moving_average.py
# A-Shares Double Moving Average Crossover Strategy
# This strategy buys when the short MA crosses above the long MA (golden cross)
# and sells when the short MA crosses below the long MA (death cross)
# Considers T+1 trading rule in A-shares market

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class DoubleMAStrategy:
    def __init__(self, short_window=5, long_window=20):
        """
        Initialize the strategy with MA parameters
        
        Parameters:
        short_window (int): Days for short moving average calculation
        long_window (int): Days for long moving average calculation
        """
        self.short_window = short_window
        self.long_window = long_window
        self.positions = None
        self.buy_orders_placed = {}  # Track buy orders for T+1 rule

    def generate_signals(self, data):
        """
        Generate trading signals based on moving average crossover
        
        Parameters:
        data (DataFrame): DataFrame with 'date' and 'close' columns
        
        Returns:
        DataFrame: Original data with signal columns added
        """
        # Make a copy of the input data
        signals = data.copy()
        
        # Create moving averages
        signals['short_ma'] = signals['close'].rolling(window=self.short_window).mean()
        signals['long_ma'] = signals['close'].rolling(window=self.long_window).mean()
        
        # Create signals
        signals['signal'] = 0.0
        signals['position'] = 0.0
        
        # Generate buy/sell signals
        signals['signal'] = np.where(signals['short_ma'] > signals['long_ma'], 1.0, 0.0)
        
        # Account for T+1 rule - can only sell shares bought before today
        signals['position'] = signals['signal'].diff()
        
        # Apply T+1 rule: can't sell on the same day as buy
        self.positions = pd.DataFrame(index=signals.index)
        self.positions['available_to_sell'] = 0
        
        for i in range(1, len(signals)):
            date = signals.index[i]
            prev_date = signals.index[i-1]
            
            # If buy signal
            if signals['position'].iloc[i] > 0:
                self.buy_orders_placed[date] = 1
                signals.at[signals.index[i], 'position'] = 1  # Buy signal
            
            # If sell signal and have available positions
            elif signals['position'].iloc[i] < 0:
                # Check if we have shares available to sell (bought before today)
                available = sum([v for k, v in self.buy_orders_placed.items() 
                               if k < date - timedelta(days=1)])
                
                if available > 0:
                    signals.at[signals.index[i], 'position'] = -1  # Sell signal
                    # Remove the oldest buy order
                    oldest_key = min([k for k in self.buy_orders_placed.keys() 
                                    if k < date - timedelta(days=1)])
                    self.buy_orders_placed.pop(oldest_key)
                else:
                    signals.at[signals.index[i], 'position'] = 0  # Can't sell, no position
            
            self.positions.at[date, 'available_to_sell'] = sum([v for k, v in 
                                                             self.buy_orders_placed.items() 
                                                             if k < date - timedelta(days=1)])
        
        return signals
    
    def backtest(self, data, initial_capital=100000.0):
        """
        Backtest the strategy
        
        Parameters:
        data (DataFrame): DataFrame with 'date' and 'close' columns
        initial_capital (float): Starting capital for backtest
        
        Returns:
        DataFrame: Backtest results including portfolio values
        """
        # Generate signals
        signals = self.generate_signals(data)
        
        # Create portfolio dataframe
        portfolio = signals.copy()
        portfolio['holdings'] = 0.0
        portfolio['cash'] = initial_capital
        portfolio['total'] = initial_capital
        
        # Apply price limit rules (±10% for A-shares)
        portfolio['price_limit_up'] = portfolio['close'].shift(1) * 1.1
        portfolio['price_limit_down'] = portfolio['close'].shift(1) * 0.9
        portfolio['tradable_price'] = portfolio['close'].copy()
        
        # Limit price to price limit rules
        portfolio.loc[portfolio['tradable_price'] > portfolio['price_limit_up'], 'tradable_price'] = portfolio['price_limit_up']
        portfolio.loc[portfolio['tradable_price'] < portfolio['price_limit_down'], 'tradable_price'] = portfolio['price_limit_down']
        
        # Transaction costs (commission: 0.05%, stamp tax: 0.1% for sell only)
        commission_rate = 0.0005
        stamp_tax_rate = 0.001
        
        # Track positions and capital
        shares = 0
        for i in range(1, len(portfolio)):
            # No trade
            if portfolio['position'].iloc[i] == 0:
                portfolio.at[portfolio.index[i], 'holdings'] = shares * portfolio['tradable_price'].iloc[i]
                portfolio.at[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
            
            # Buy signal
            elif portfolio['position'].iloc[i] > 0:
                # Calculate max shares buyable with cash
                max_shares = int(portfolio['cash'].iloc[i-1] / 
                              (portfolio['tradable_price'].iloc[i] * (1 + commission_rate)))
                
                # Buy 100 shares at a time (A-shares rule)
                buy_shares = (max_shares // 100) * 100
                
                if buy_shares > 0:
                    cost = buy_shares * portfolio['tradable_price'].iloc[i]
                    commission = cost * commission_rate
                    shares += buy_shares
                    
                    portfolio.at[portfolio.index[i], 'holdings'] = shares * portfolio['tradable_price'].iloc[i]
                    portfolio.at[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] - cost - commission
                else:
                    portfolio.at[portfolio.index[i], 'holdings'] = shares * portfolio['tradable_price'].iloc[i]
                    portfolio.at[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
            
            # Sell signal
            elif portfolio['position'].iloc[i] < 0 and shares > 0:
                # Available to sell considering T+1 rule
                available_shares = min(shares, self.positions.iloc[i]['available_to_sell'] * 100)
                
                # Sell 100 shares at a time
                sell_shares = (available_shares // 100) * 100
                
                if sell_shares > 0:
                    revenue = sell_shares * portfolio['tradable_price'].iloc[i]
                    commission = revenue * commission_rate
                    stamp_tax = revenue * stamp_tax_rate
                    shares -= sell_shares
                    
                    portfolio.at[portfolio.index[i], 'holdings'] = shares * portfolio['tradable_price'].iloc[i]
                    portfolio.at[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] + revenue - commission - stamp_tax
                else:
                    portfolio.at[portfolio.index[i], 'holdings'] = shares * portfolio['tradable_price'].iloc[i]
                    portfolio.at[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
            
            portfolio.at[portfolio.index[i], 'total'] = portfolio['holdings'].iloc[i] + portfolio['cash'].iloc[i]
        
        # Calculate returns
        portfolio['returns'] = portfolio['total'].pct_change()
        portfolio['strategy_returns'] = portfolio['total'] / portfolio['total'].iloc[0] - 1
        portfolio['market_returns'] = portfolio['close'] / portfolio['close'].iloc[0] - 1
        
        return portfolio

    def plot_results(self, results):
        """
        Plot the backtest results
        
        Parameters:
        results (DataFrame): Backtest results from backtest method
        """
        plt.figure(figsize=(12, 9))
        
        # Plot portfolio value
        plt.subplot(311)
        plt.plot(results['total'])
        plt.title('Portfolio Value')
        plt.grid(True)
        
        # Plot strategy vs market returns
        plt.subplot(312)
        plt.plot(results['strategy_returns'], label='Strategy')
        plt.plot(results['market_returns'], label='Market')
        plt.title('Strategy vs Market Returns')
        plt.legend()
        plt.grid(True)
        
        # Plot moving averages and buy/sell signals
        plt.subplot(313)
        plt.plot(results['close'], label='Price')
        plt.plot(results['short_ma'], label=f'{self.short_window}-day MA')
        plt.plot(results['long_ma'], label=f'{self.long_window}-day MA')
        
        # Plot buy signals
        plt.plot(results.loc[results['position'] == 1].index, 
                results.loc[results['position'] == 1]['close'], 
                '^', markersize=10, color='g', label='Buy')
        
        # Plot sell signals
        plt.plot(results.loc[results['position'] == -1].index, 
                results.loc[results['position'] == -1]['close'], 
                'v', markersize=10, color='r', label='Sell')
        
        plt.title('Moving Averages and Signals')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Import tushare for A-shares data
    import tushare as ts
    
    # Set your tushare token
    # ts.set_token('YOUR_TUSHARE_TOKEN')
    
    # Example: Get data for a stock
    # df = ts.pro_bar(ts_code='000001.SZ', start_date='20210101', end_date='20211231')
    # df['date'] = pd.to_datetime(df['trade_date'])
    # df.set_index('date', inplace=True)
    # df.sort_index(inplace=True)
    
    # For demonstration, create dummy data
    dates = pd.date_range(start='2021-01-01', end='2021-12-31')
    np.random.seed(42)
    closes = np.random.normal(100, 2, len(dates)).cumsum() + 100
    
    # Create test dataframe
    df = pd.DataFrame({
        'close': closes
    }, index=dates)
    
    # Initialize strategy
    strategy = DoubleMAStrategy(short_window=5, long_window=20)
    
    # Backtest
    results = strategy.backtest(df)
    
    # Plot results
    # strategy.plot_results(results)
    
    # Print key metrics
    final_value = results['total'].iloc[-1]
    returns = (final_value / 100000 - 1) * 100
    print(f"Final Portfolio Value: {final_value:.2f}")
    print(f"Total Return: {returns:.2f}%")
    
    # Calculate Sharpe Ratio (annualized)
    risk_free_rate = 0.03 / 252  # Daily risk-free rate (assuming 3% annual)
    excess_returns = results['returns'] - risk_free_rate
    sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Calculate Maximum Drawdown
    cumulative_returns = (1 + results['returns']).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max - 1) * 100
    max_drawdown = drawdown.min()
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
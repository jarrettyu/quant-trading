# rsi_strategy.py
# A-Shares RSI (Relative Strength Index) Strategy
# Buy when RSI is below oversold threshold (30)
# Sell when RSI is above overbought threshold (70)
# Implements A-shares specific rules like T+1 and price limits

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class RSIStrategy:
    def __init__(self, rsi_period=14, oversold=30, overbought=70):
        """
        Initialize the RSI strategy
        
        Parameters:
        rsi_period (int): Period for RSI calculation
        oversold (int): RSI level to trigger buy signals
        overbought (int): RSI level to trigger sell signals
        """
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.buy_orders_placed = {}  # Track buy orders for T+1 rule
        
    def calculate_rsi(self, data, window=14):
        """
        Calculate Relative Strength Index
        
        Parameters:
        data (Series): Price series
        window (int): RSI calculation period
        
        Returns:
        Series: RSI values
        """
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, data):
        """
        Generate trading signals based on RSI
        
        Parameters:
        data (DataFrame): DataFrame with 'date' and 'close' columns
        
        Returns:
        DataFrame: Original data with signal columns added
        """
        # Make a copy of the input data
        signals = data.copy()
        
        # Calculate RSI
        signals['rsi'] = self.calculate_rsi(signals['close'], self.rsi_period)
        
        # Initialize signal and position columns
        signals['signal'] = 0.0
        signals['position'] = 0.0
        
        # Generate buy signal when RSI crosses below oversold level
        signals.loc[signals['rsi'] < self.oversold, 'signal'] = 1.0
        
        # Generate sell signal when RSI crosses above overbought level
        signals.loc[signals['rsi'] > self.overbought, 'signal'] = 0.0
        
        # Apply T+1 rule: can only sell shares bought before today
        positions = []
        available_to_sell = 0
        
        for i in range(len(signals)):
            date = signals.index[i]
            
            # If RSI indicates buy
            if signals['rsi'].iloc[i] < self.oversold:
                if i > 0 and signals['signal'].iloc[i-1] != 1.0:  # New buy signal
                    self.buy_orders_placed[date] = 1
                    positions.append(1)  # Buy signal
                else:
                    positions.append(0)  # Hold
            
            # If RSI indicates sell and we have positions to sell
            elif signals['rsi'].iloc[i] > self.overbought:
                # Calculate available positions (bought at least 1 day ago)
                available_to_sell = sum([v for k, v in self.buy_orders_placed.items() 
                                      if k < date - timedelta(days=1)])
                
                if available_to_sell > 0:
                    # Remove oldest buy order
                    oldest_keys = [k for k in self.buy_orders_placed.keys() 
                                 if k < date - timedelta(days=1)]
                    if oldest_keys:
                        oldest_key = min(oldest_keys)
                        self.buy_orders_placed.pop(oldest_key)
                    
                    positions.append(-1)  # Sell signal
                else:
                    positions.append(0)  # Want to sell but can't due to T+1
            else:
                positions.append(0)  # No signal
        
        signals['position'] = positions
        
        return signals
    
    def backtest(self, data, initial_capital=100000.0):
        """
        Backtest the RSI strategy
        
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
                # Sell 100 shares at a time
                sell_shares = min(shares, 100)
                
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
        
        # Plot RSI and buy/sell signals
        plt.subplot(313)
        plt.plot(results['rsi'], label='RSI')
        plt.axhline(y=self.overbought, color='r', linestyle='--', label='Overbought')
        plt.axhline(y=self.oversold, color='g', linestyle='--', label='Oversold')
        
        # Plot buy signals
        plt.plot(results.loc[results['position'] == 1].index, 
                results.loc[results['position'] == 1]['rsi'], 
                '^', markersize=10, color='g', label='Buy')
        
        # Plot sell signals
        plt.plot(results.loc[results['position'] == -1].index, 
                results.loc[results['position'] == -1]['rsi'], 
                'v', markersize=10, color='r', label='Sell')
        
        plt.title('RSI and Signals')
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
    
    # Add some mean-reverting behavior for RSI demonstration
    for i in range(1, len(closes)):
        if closes[i-1] > 120:
            closes[i] = closes[i-1] - np.random.uniform(1, 3)
        elif closes[i-1] < 80:
            closes[i] = closes[i-1] + np.random.uniform(1, 3)
    
    # Create test dataframe
    df = pd.DataFrame({
        'close': closes
    }, index=dates)
    
    # Initialize strategy
    strategy = RSIStrategy(rsi_period=14, oversold=30, overbought=70)
    
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
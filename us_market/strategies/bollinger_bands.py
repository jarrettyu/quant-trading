# bollinger_bands.py
# US Market Bollinger Bands Mean Reversion Strategy
# This strategy buys when price touches the lower Bollinger Band
# and sells when price touches the upper Bollinger Band

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class BollingerBandsStrategy:
    def __init__(self, window=20, num_std=2):
        """
        Initialize the Bollinger Bands strategy
        
        Parameters:
        window (int): Moving average window for Bollinger Bands
        num_std (int): Number of standard deviations for bands
        """
        self.window = window
        self.num_std = num_std
    
    def calculate_bollinger_bands(self, data):
        """
        Calculate Bollinger Bands
        
        Parameters:
        data (Series): Price series
        
        Returns:
        DataFrame: DataFrame with middle band, upper band, and lower band
        """
        # Calculate middle band (SMA)
        middle_band = data.rolling(window=self.window).mean()
        
        # Calculate standard deviation
        rolling_std = data.rolling(window=self.window).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (rolling_std * self.num_std)
        lower_band = middle_band - (rolling_std * self.num_std)
        
        # Create and return bollinger bands DataFrame
        bollinger_bands = pd.DataFrame({
            'middle_band': middle_band,
            'upper_band': upper_band,
            'lower_band': lower_band
        })
        
        return bollinger_bands
    
    def generate_signals(self, data):
        """
        Generate trading signals based on Bollinger Bands
        
        Parameters:
        data (DataFrame): DataFrame with 'close' column
        
        Returns:
        DataFrame: Original data with signal columns added
        """
        # Make a copy of the input data
        signals = data.copy()
        
        # Calculate Bollinger Bands
        bollinger_bands = self.calculate_bollinger_bands(signals['close'])
        signals = pd.concat([signals, bollinger_bands], axis=1)
        
        # Calculate price position relative to bands
        signals['pct_b'] = (signals['close'] - signals['lower_band']) / (signals['upper_band'] - signals['lower_band'])
        
        # Initialize signal and position columns
        signals['signal'] = 0
        signals['position'] = 0
        
        # Generate signals
        for i in range(1, len(signals)):
            # Buy signal: price crosses below lower band
            if signals['close'].iloc[i] <= signals['lower_band'].iloc[i] and signals['close'].iloc[i-1] > signals['lower_band'].iloc[i-1]:
                signals.at[signals.index[i], 'signal'] = 1
            
            # Sell signal: price crosses above upper band
            elif signals['close'].iloc[i] >= signals['upper_band'].iloc[i] and signals['close'].iloc[i-1] < signals['upper_band'].iloc[i-1]:
                signals.at[signals.index[i], 'signal'] = -1
            
            # Position: cumulative signals
            if signals['signal'].iloc[i] == 1:  # Buy signal
                signals.at[signals.index[i], 'position'] = 1
            elif signals['signal'].iloc[i] == -1:  # Sell signal
                signals.at[signals.index[i], 'position'] = 0
            else:  # No signal
                signals.at[signals.index[i], 'position'] = signals['position'].iloc[i-1]
        
        return signals
    
    def backtest(self, data, initial_capital=100000.0):
        """
        Backtest the Bollinger Bands strategy
        
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
        
        # Transaction costs (commission: 0.1%)
        commission_rate = 0.001
        
        # Track positions and capital
        shares = 0
        for i in range(1, len(portfolio)):
            # If position changed
            if portfolio['position'].iloc[i] != portfolio['position'].iloc[i-1]:
                price = portfolio['close'].iloc[i]
                
                if portfolio['position'].iloc[i] == 1:  # Buy
                    # Calculate shares to buy with 95% of capital
                    shares_to_buy = int(portfolio['cash'].iloc[i-1] * 0.95 / (price * (1 + commission_rate)))
                    
                    # Calculate cost
                    cost = shares_to_buy * price
                    commission = cost * commission_rate
                    
                    # Update shares and cash
                    shares = shares_to_buy
                    portfolio.at[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] - cost - commission
                
                elif portfolio['position'].iloc[i] == 0 and shares > 0:  # Sell
                    # Calculate revenue
                    revenue = shares * price
                    commission = revenue * commission_rate
                    
                    # Update shares and cash
                    portfolio.at[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1] + revenue - commission
                    shares = 0
            
            else:  # No change in position
                portfolio.at[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
            
            # Update holdings and total
            portfolio.at[portfolio.index[i], 'holdings'] = shares * portfolio['close'].iloc[i]
            portfolio.at[portfolio.index[i], 'total'] = portfolio['cash'].iloc[i] + portfolio['holdings'].iloc[i]
        
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
        
        # Plot price and Bollinger Bands
        plt.subplot(313)
        plt.plot(results['close'], label='Price')
        plt.plot(results['middle_band'], label='Middle Band', linestyle='--')
        plt.plot(results['upper_band'], label='Upper Band', linestyle='--')
        plt.plot(results['lower_band'], label='Lower Band', linestyle='--')
        
        # Plot buy signals
        plt.plot(results.loc[results['signal'] == 1].index, 
                results.loc[results['signal'] == 1]['close'], 
                '^', markersize=10, color='g', label='Buy')
        
        # Plot sell signals
        plt.plot(results.loc[results['signal'] == -1].index, 
                results.loc[results['signal'] == -1]['close'], 
                'v', markersize=10, color='r', label='Sell')
        
        plt.title('Price and Bollinger Bands')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Import yfinance for US market data
    import yfinance as yf
    
    # Example: Get data for a stock
    # ticker = "AAPL"
    # df = yf.download(ticker, start="2020-01-01", end="2021-12-31")
    
    # For demonstration, create dummy data
    dates = pd.date_range(start='2021-01-01', end='2021-12-31')
    np.random.seed(42)
    
    # Generate mean-reverting price series for Bollinger Bands demonstration
    price = 100
    prices = [price]
    for _ in range(1, len(dates)):
        # Mean reversion component
        mean_reversion = 0.1 * (100 - price)
        # Random component
        random_component = np.random.normal(0, 1)
        # Update price
        price = price + mean_reversion + random_component
        prices.append(price)
    
    # Create test dataframe
    df = pd.DataFrame({
        'close': prices,
        'open': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
        'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        'volume': np.random.uniform(1e6, 1e7, len(dates))
    }, index=dates)
    
    # Initialize strategy
    strategy = BollingerBandsStrategy(window=20, num_std=2)
    
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
    risk_free_rate = 0.02 / 252  # Daily risk-free rate (assuming 2% annual)
    excess_returns = results['returns'] - risk_free_rate
    sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Calculate Maximum Drawdown
    cumulative_returns = (1 + results['returns']).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max - 1) * 100
    max_drawdown = drawdown.min()
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    
    # Compare to buy and hold
    buy_hold_return = (results['close'].iloc[-1] / results['close'].iloc[0] - 1) * 100
    print(f"Buy and Hold Return: {buy_hold_return:.2f}%")
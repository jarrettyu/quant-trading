# northbound_flow.py
# Hong Kong Northbound Flow Strategy
# This strategy tracks capital flows from mainland China into Hong Kong
# through the Stock Connect program and buys stocks with significant inflows

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class NorthboundFlowStrategy:
    def __init__(self, flow_threshold=0.2, holding_period=5, top_n=5):
        """
        Initialize the Northbound Flow strategy
        
        Parameters:
        flow_threshold (float): Percentage of market cap to consider significant (0.2%)
        holding_period (int): Number of days to hold positions
        top_n (int): Number of top stocks to include in portfolio
        """
        self.flow_threshold = flow_threshold  # Flow threshold as percentage of market cap
        self.holding_period = holding_period  # Holding period in days
        self.top_n = top_n                    # Number of stocks to include
        self.positions = {}                   # Dictionary to track positions
        
    def generate_signals(self, data):
        """
        Generate trading signals based on northbound flows
        
        Parameters:
        data (DataFrame): DataFrame with stock data and northbound flows
        
        Returns:
        DataFrame: Original data with signal columns added
        """
        # Group by date and ticker
        grouped = data.groupby(['date', 'ticker'])
        
        # Initialize results
        signals = []
        
        # Process each date
        for date, group in data.groupby('date'):
            # Calculate flow percentage (flow / market_cap)
            group['flow_percentage'] = group['northbound_flow'] / group['market_cap'] * 100
            
            # Sort stocks by flow percentage
            sorted_stocks = group.sort_values('flow_percentage', ascending=False)
            
            # Select top N stocks with flows above threshold
            top_stocks = sorted_stocks[sorted_stocks['flow_percentage'] > self.flow_threshold].head(self.top_n)
            
            # Generate buy signals for top stocks
            for idx, stock in top_stocks.iterrows():
                # Check if we should exit any positions
                for ticker, entry_date in list(self.positions.items()):
                    hold_days = (date - entry_date).days
                    if hold_days >= self.holding_period:
                        # Exit position
                        exit_row = stock.copy()
                        exit_row['ticker'] = ticker
                        exit_row['signal'] = -1
                        signals.append(exit_row)
                        del self.positions[ticker]
                
                # New buy signal
                if stock['ticker'] not in self.positions:
                    buy_row = stock.copy()
                    buy_row['signal'] = 1
                    signals.append(buy_row)
                    self.positions[stock['ticker']] = date
                else:
                    # Already holding
                    hold_row = stock.copy()
                    hold_row['signal'] = 0
                    signals.append(hold_row)
            
            # Add hold signals for existing positions
            for ticker, entry_date in self.positions.items():
                if ticker not in top_stocks['ticker'].values:
                    # Check if ticker exists in the current date's data
                    if ticker in group['ticker'].values:
                        hold_row = group[group['ticker'] == ticker].iloc[0].copy()
                        hold_row['signal'] = 0
                        signals.append(hold_row)
        
        # Convert to DataFrame
        signals_df = pd.DataFrame(signals)
        
        return signals_df
    
    def backtest(self, data, initial_capital=1000000.0):
        """
        Backtest the Northbound Flow strategy
        
        Parameters:
        data (DataFrame): DataFrame with stock data and northbound flows
        initial_capital (float): Starting capital for backtest
        
        Returns:
        tuple: (portfolio DataFrame, positions DataFrame)
        """
        # Generate signals
        signals = self.generate_signals(data)
        
        # Initialize portfolio
        portfolio = pd.DataFrame(index=pd.unique(data['date']))
        portfolio['cash'] = initial_capital
        portfolio['holdings'] = 0.0
        portfolio['total'] = initial_capital
        
        # Initialize positions dataframe
        positions = pd.DataFrame(columns=['ticker', 'shares', 'entry_price', 'entry_date', 'exit_date', 'exit_price', 'return'])
        
        # Transaction costs
        commission_rate = 0.0007  # HK commission: 0.07%
        stamp_tax_rate = 0.001    # HK stamp tax: 0.1%
        
        # Track current positions
        current_positions = {}  # ticker -> (shares, entry_price, entry_date)
        
        # Process signals chronologically
        for date, day_signals in signals.groupby('date'):
            # Handle buy signals
            buy_signals = day_signals[day_signals['signal'] == 1]
            if not buy_signals.empty:
                # Allocate capital equally among buy signals
                allocation_per_stock = portfolio.loc[date, 'cash'] / len(buy_signals) if len(buy_signals) > 0 else 0
                
                for _, signal in buy_signals.iterrows():
                    ticker = signal['ticker']
                    price = signal['close']
                    
                    # Calculate shares to buy (round to nearest lot size, typically 500 for HK)
                    shares_to_buy = int((allocation_per_stock * 0.95) // (price * (1 + commission_rate) * 500)) * 500
                    
                    if shares_to_buy > 0:
                        cost = shares_to_buy * price
                        commission = cost * commission_rate
                        
                        # Update cash
                        portfolio.loc[date, 'cash'] -= (cost + commission)
                        
                        # Record position
                        current_positions[ticker] = (shares_to_buy, price, date)
                
            # Handle sell signals
            sell_signals = day_signals[day_signals['signal'] == -1]
            for _, signal in sell_signals.iterrows():
                ticker = signal['ticker']
                if ticker in current_positions:
                    shares, entry_price, entry_date = current_positions[ticker]
                    exit_price = signal['close']
                    
                    # Calculate revenue
                    revenue = shares * exit_price
                    commission = revenue * commission_rate
                    stamp_tax = revenue * stamp_tax_rate
                    
                    # Update cash
                    portfolio.loc[date, 'cash'] += (revenue - commission - stamp_tax)
                    
                    # Record trade in positions dataframe
                    position_return = (exit_price / entry_price - 1) * 100
                    new_position = pd.DataFrame({
                        'ticker': [ticker],
                        'shares': [shares],
                        'entry_price': [entry_price],
                        'entry_date': [entry_date],
                        'exit_date': [date],
                        'exit_price': [exit_price],
                        'return': [position_return]
                    })
                    positions = pd.concat([positions, new_position], ignore_index=True)
                    
                    # Remove from current positions
                    del current_positions[ticker]
            
            # Calculate holdings value
            holdings_value = 0
            for ticker, (shares, _, _) in current_positions.items():
                # Get current price
                current_price = day_signals[day_signals['ticker'] == ticker]['close'].iloc[0] if ticker in day_signals['ticker'].values else 0
                holdings_value += shares * current_price
            
            portfolio.loc[date, 'holdings'] = holdings_value
            portfolio.loc[date, 'total'] = portfolio.loc[date, 'cash'] + portfolio.loc[date, 'holdings']
        
        # Fill forward for days without signals
        portfolio = portfolio.fillna(method='ffill')
        
        # Calculate returns
        portfolio['returns'] = portfolio['total'].pct_change()
        portfolio['strategy_returns'] = portfolio['total'] / portfolio['total'].iloc[0] - 1
        
        # Create a benchmark (average of all stocks or index)
        # This is simplified; in practice, you'd use a relevant index like Hang Seng
        benchmark = data.groupby('date')['close'].mean()
        portfolio['benchmark'] = benchmark / benchmark.iloc[0]
        portfolio['benchmark_returns'] = portfolio['benchmark'] - 1
        
        return portfolio, positions
    
    def plot_results(self, portfolio, positions):
        """
        Plot the backtest results
        
        Parameters:
        portfolio (DataFrame): Portfolio values over time
        positions (DataFrame): Individual position details
        """
        plt.figure(figsize=(12, 9))
        
        # Plot portfolio value
        plt.subplot(211)
        plt.plot(portfolio.index, portfolio['total'])
        plt.title('Portfolio Value')
        plt.grid(True)
        
        # Plot strategy vs benchmark returns
        plt.subplot(212)
        plt.plot(portfolio.index, portfolio['strategy_returns']*100, label='Strategy')
        plt.plot(portfolio.index, portfolio['benchmark_returns']*100, label='Benchmark')
        plt.title('Strategy vs Benchmark Returns (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Plot position returns
        if not positions.empty:
            plt.figure(figsize=(10, 6))
            positions.sort_values('return').plot(kind='barh', x='ticker', y='return')
            plt.title('Returns by Position (%)')
            plt.grid(True)
            plt.tight_layout()
            plt.show()


# Example usage
if __name__ == "__main__":
    # For demonstration, create dummy data
    # In practice, would use real northbound flow data from providers like Wind, Bloomberg, etc.
    
    # Generate dates
    dates = pd.date_range(start='2021-01-01', end='2021-03-31', freq='B')  # Business days
    
    # Generate tickers
    tickers = ['0700.HK', '0941.HK', '1398.HK', '0388.HK', '2318.HK', 
              '0883.HK', '0939.HK', '2628.HK', '0386.HK', '0016.HK']
    
    # Create empty dataframe
    data_rows = []
    
    np.random.seed(42)
    
    # Generate data for each date and ticker
    for date in dates:
        for ticker in tickers:
            # Base price around 100 with some random variation by ticker
            ticker_base = 50 + hash(ticker) % 100
            
            # Create price with some trend and randomness
            price_offset = (date - dates[0]).days / 30  # Trend component
            price = ticker_base * (1 + price_offset * 0.05) * (1 + np.random.normal(0, 0.02))
            
            # Create market cap
            shares_outstanding = np.random.uniform(1e9, 5e9)
            market_cap = price * shares_outstanding
            
            # Create northbound flow (mostly small, occasionally large)
            flow_base = market_cap * np.random.normal(0, 0.001)  # Mostly small flows
            
            # Add occasional large flows (5% chance)
            if np.random.random() < 0.05:
                flow_base = market_cap * np.random.uniform(0.002, 0.005)  # Larger flow
            
            # Create row
            row = {
                'date': date,
                'ticker': ticker,
                'open': price * (1 - np.random.uniform(0, 0.01)),
                'high': price * (1 + np.random.uniform(0, 0.02)),
                'low': price * (1 - np.random.uniform(0, 0.02)),
                'close': price,
                'volume': np.random.uniform(1e6, 1e7),
                'market_cap': market_cap,
                'northbound_flow': flow_base
            }
            
            data_rows.append(row)
    
    # Create dataframe
    df = pd.DataFrame(data_rows)
    
    # Initialize strategy
    strategy = NorthboundFlowStrategy(flow_threshold=0.2, holding_period=5, top_n=3)
    
    # Backtest
    portfolio, positions = strategy.backtest(df)
    
    # Plot results
    # strategy.plot_results(portfolio, positions)
    
    # Print key metrics
    final_value = portfolio['total'].iloc[-1]
    initial_value = portfolio['total'].iloc[0]
    returns = (final_value / initial_value - 1) * 100
    print(f"Final Portfolio Value: {final_value:.2f} HKD")
    print(f"Total Return: {returns:.2f}%")
    
    # Calculate Sharpe Ratio (annualized)
    risk_free_rate = 0.01 / 252  # Daily risk-free rate (assuming 1% annual)
    excess_returns = portfolio['returns'] - risk_free_rate
    sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Calculate Maximum Drawdown
    cumulative_returns = (1 + portfolio['returns']).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max - 1) * 100
    max_drawdown = drawdown.min()
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    
    # Print position statistics
    if not positions.empty:
        print(f"\nTotal Positions: {len(positions)}")
        print(f"Average Position Return: {positions['return'].mean():.2f}%")
        print(f"Win Rate: {(positions['return'] > 0).mean()*100:.2f}%")
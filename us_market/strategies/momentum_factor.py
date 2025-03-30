# momentum_factor.py
# US Market Momentum Factor Strategy
# This strategy buys stocks with the strongest price momentum
# and rebalances the portfolio periodically

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class MomentumFactorStrategy:
    def __init__(self, lookback_period=90, holding_period=30, top_n=10):
        """
        Initialize the Momentum Factor strategy
        
        Parameters:
        lookback_period (int): Days to look back for momentum calculation
        holding_period (int): Days to hold positions before rebalancing
        top_n (int): Number of top stocks to include in portfolio
        """
        self.lookback_period = lookback_period
        self.holding_period = holding_period
        self.top_n = top_n
    
    def calculate_momentum(self, prices):
        """
        Calculate momentum for each stock
        
        Parameters:
        prices (DataFrame): DataFrame with dates as index and tickers as columns
        
        Returns:
        Series: Momentum for each ticker
        """
        # Calculate returns over the lookback period
        returns = prices.pct_change(self.lookback_period)
        
        # Get the latest returns
        latest_returns = returns.iloc[-1]
        
        return latest_returns
    
    def generate_signals(self, prices):
        """
        Generate trading signals based on momentum
        
        Parameters:
        prices (DataFrame): DataFrame with dates as index and tickers as columns
        
        Returns:
        DataFrame: DataFrame with rebalance dates and portfolio weights
        """
        # Initialize results
        all_weights = pd.DataFrame(index=prices.index, columns=prices.columns)
        all_weights = all_weights.fillna(0)
        
        # Define rebalance dates (start date and every holding_period days)
        start_date = prices.index[self.lookback_period]
        rebalance_dates = [start_date]
        
        current_date = start_date
        while current_date < prices.index[-1]:
            current_date = prices.loc[prices.index > current_date].index[min(self.holding_period, len(prices.loc[prices.index > current_date].index) - 1)]
            rebalance_dates.append(current_date)
        
        # Generate signals for each rebalance date
        for date in rebalance_dates:
            # Get historical data up to rebalance date
            hist_prices = prices.loc[prices.index <= date]
            
            # Skip if not enough history
            if len(hist_prices) <= self.lookback_period:
                continue
            
            # Calculate momentum
            momentum = self.calculate_momentum(hist_prices)
            
            # Select top N stocks
            top_stocks = momentum.nlargest(self.top_n)
            
            # Equal weight portfolio
            weights = top_stocks * 0 + 1 / len(top_stocks)
            
            # Set weights
            all_weights.loc[date, top_stocks.index] = weights.values
        
        # Forward fill weights between rebalance dates
        all_weights = all_weights.fillna(method='ffill')
        
        return all_weights
    
    def backtest(self, prices, initial_capital=100000.0):
        """
        Backtest the Momentum Factor strategy
        
        Parameters:
        prices (DataFrame): DataFrame with dates as index and tickers as columns
        initial_capital (float): Starting capital for backtest
        
        Returns:
        tuple: (portfolio DataFrame, weights DataFrame)
        """
        # Generate weights
        weights = self.generate_signals(prices)
        
        # Calculate daily returns for each stock
        returns = prices.pct_change().fillna(0)
        
        # Calculate strategy returns
        strategy_returns = (weights.shift(1) * returns).sum(axis=1)
        
        # Transaction costs (0.1% one-way)
        transaction_cost = 0.001
        
        # Calculate turnover and transaction costs
        weight_changes = weights.diff().abs().sum(axis=1)
        transaction_costs = weight_changes * transaction_cost
        
        # Adjust returns for transaction costs
        strategy_returns = strategy_returns - transaction_costs
        
        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # Calculate portfolio value
        portfolio_value = initial_capital * cumulative_returns
        
        # Create portfolio DataFrame
        portfolio = pd.DataFrame({
            'returns': strategy_returns,
            'cumulative_returns': cumulative_returns,
            'portfolio_value': portfolio_value
        })
        
        return portfolio, weights
    
    def plot_results(self, portfolio, weights, benchmark=None):
        """
        Plot the backtest results
        
        Parameters:
        portfolio (DataFrame): Portfolio DataFrame from backtest
        weights (DataFrame): Weights DataFrame from backtest
        benchmark (Series, optional): Benchmark returns
        """
        plt.figure(figsize=(12, 9))
        
        # Plot portfolio value
        plt.subplot(211)
        plt.plot(portfolio['portfolio_value'])
        plt.title('Portfolio Value')
        plt.grid(True)
        
        # Plot strategy vs benchmark returns
        plt.subplot(212)
        plt.plot(portfolio['cumulative_returns'] - 1, label='Strategy')
        
        if benchmark is not None:
            benchmark_returns = (1 + benchmark).cumprod() - 1
            plt.plot(benchmark_returns, label='Benchmark')
        
        plt.title('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Plot portfolio weights over time
        plt.figure(figsize=(12, 6))
        plt.stackplot(weights.index, weights.T.values, labels=weights.columns)
        plt.title('Portfolio Weights Over Time')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Import yfinance for US market data
    import yfinance as yf
    
    # Example: Get data for multiple stocks
    # tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "NVDA", "JPM", "V", "JNJ", 
    #           "WMT", "PG", "MA", "UNH", "HD", "BAC", "INTC", "VZ", "ADBE", "CRM"]
    # 
    # start_date = "2018-01-01"
    # end_date = "2021-12-31"
    # 
    # data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # For demonstration, create dummy data
    dates = pd.date_range(start='2020-01-01', end='2021-12-31')
    np.random.seed(42)
    
    # Create 20 simulated stocks
    n_stocks = 20
    stocks = [f'STOCK_{i}' for i in range(1, n_stocks + 1)]
    
    # Create price data with different momentum characteristics
    price_data = {}
    
    for i, stock in enumerate(stocks):
        # Base price around 100 with some random variation by stock
        base_price = 100 * (1 + i * 0.02)
        
        # Create price trend (some stocks with strong momentum, others with weak)
        momentum_factor = np.random.uniform(-0.1, 0.3)  # Yearly return between -10% and +30%
        daily_drift = momentum_factor / 252  # Convert to daily
        
        # Add volatility
        volatility = np.random.uniform(0.01, 0.03)  # Annual volatility between 10% and 30%
        daily_volatility = volatility / np.sqrt(252)  # Convert to daily
        
        # Generate prices
        prices = [base_price]
        for t in range(1, len(dates)):
            # Random component
            random_return = np.random.normal(daily_drift, daily_volatility)
            
            # New price
            new_price = prices[-1] * (1 + random_return)
            prices.append(new_price)
        
        price_data[stock] = prices
    
    # Create DataFrame
    df = pd.DataFrame(price_data, index=dates)
    
    # Create SPY-like benchmark
    benchmark_drift = 0.12 / 252  # 12% annual return
    benchmark_vol = 0.16 / np.sqrt(252)  # 16% annual volatility
    
    benchmark_prices = [100]
    for t in range(1, len(dates)):
        benchmark_return = np.random.normal(benchmark_drift, benchmark_vol)
        benchmark_prices.append(benchmark_prices[-1] * (1 + benchmark_return))
    
    benchmark = pd.Series(benchmark_prices, index=dates)
    benchmark_returns = benchmark.pct_change().fillna(0)
    
    # Initialize strategy
    strategy = MomentumFactorStrategy(lookback_period=60, holding_period=20, top_n=5)
    
    # Backtest
    portfolio, weights = strategy.backtest(df)
    
    # Plot results
    # strategy.plot_results(portfolio, weights, benchmark_returns)
    
    # Print key metrics
    final_value = portfolio['portfolio_value'].iloc[-1]
    initial_value = portfolio['portfolio_value'].iloc[0]
    total_return = (final_value / initial_value - 1) * 100
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    
    # Calculate annualized return
    years = (dates[-1] - dates[0]).days / 365
    annualized_return = ((final_value / initial_value) ** (1 / years) - 1) * 100
    print(f"Annualized Return: {annualized_return:.2f}%")
    
    # Calculate Sharpe Ratio (annualized)
    risk_free_rate = 0.02 / 252  # Daily risk-free rate (assuming 2% annual)
    excess_returns = portfolio['returns'] - risk_free_rate
    sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Calculate Maximum Drawdown
    cumulative_returns = portfolio['cumulative_returns']
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max - 1) * 100
    max_drawdown = drawdown.min()
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    
    # Calculate benchmark metrics
    benchmark_final = benchmark.iloc[-1]
    benchmark_initial = benchmark.iloc[0]
    benchmark_return = (benchmark_final / benchmark_initial - 1) * 100
    print(f"\nBenchmark Return: {benchmark_return:.2f}%")
    
    # Calculate benchmark annualized return
    benchmark_annualized = ((benchmark_final / benchmark_initial) ** (1 / years) - 1) * 100
    print(f"Benchmark Annualized Return: {benchmark_annualized:.2f}%")
    
    # Calculate Information Ratio
    tracking_error = (portfolio['returns'] - benchmark_returns).std() * np.sqrt(252)
    information_ratio = (annualized_return - benchmark_annualized) / (tracking_error * 100)
    print(f"Information Ratio: {information_ratio:.2f}")
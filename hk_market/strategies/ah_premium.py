# ah_premium.py
# Hong Kong AH Premium Strategy
# This strategy exploits the price difference between A-shares and H-shares
# of the same company, buying the undervalued one and selling the overvalued one
# when the premium/discount reaches extreme levels

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class AHPremiumStrategy:
    def __init__(self, entry_threshold=20, exit_threshold=5):
        """
        Initialize the AH Premium strategy
        
        Parameters:
        entry_threshold (float): Premium percentage to trigger trades
        exit_threshold (float): Premium percentage to exit trades
        """
        self.entry_threshold = entry_threshold  # Premium threshold to enter position
        self.exit_threshold = exit_threshold    # Premium threshold to exit position
        self.positions = {}                     # Track positions
        
    def calculate_premium(self, a_price, h_price, exchange_rate):
        """
        Calculate AH premium
        
        Parameters:
        a_price (float): A-share price in CNY
        h_price (float): H-share price in HKD
        exchange_rate (float): CNY/HKD exchange rate
        
        Returns:
        float: AH premium percentage
        """
        # Convert H-share price to CNY
        h_price_cny = h_price * exchange_rate
        
        # Calculate premium
        premium = (a_price / h_price_cny - 1) * 100
        
        return premium
    
    def generate_signals(self, data):
        """
        Generate trading signals based on AH premium
        
        Parameters:
        data (DataFrame): DataFrame with A-share and H-share prices and exchange rate
        
        Returns:
        DataFrame: Original data with signal columns added
        """
        # Make a copy of the input data
        signals = data.copy()
        
        # Calculate AH premium
        signals['ah_premium'] = self.calculate_premium(
            signals['a_price'], 
            signals['h_price'], 
            signals['exchange_rate']
        )
        
        # Initialize signal and position columns
        signals['a_signal'] = 0  # Signal for A-shares
        signals['h_signal'] = 0  # Signal for H-shares
        signals['a_position'] = 0
        signals['h_position'] = 0
        
        # Generate signals based on AH premium
        for i in range(1, len(signals)):
            premium = signals['ah_premium'].iloc[i]
            
            # Entry conditions
            if abs(premium) > self.entry_threshold:
                if premium > 0:  # A-shares overvalued, H-shares undervalued
                    signals.at[signals.index[i], 'a_signal'] = -1  # Sell A-shares
                    signals.at[signals.index[i], 'h_signal'] = 1   # Buy H-shares
                else:  # A-shares undervalued, H-shares overvalued
                    signals.at[signals.index[i], 'a_signal'] = 1   # Buy A-shares
                    signals.at[signals.index[i], 'h_signal'] = -1  # Sell H-shares
            
            # Exit conditions
            elif abs(premium) < self.exit_threshold:
                signals.at[signals.index[i], 'a_signal'] = 0
                signals.at[signals.index[i], 'h_signal'] = 0
                
            # Calculate positions
            if signals['a_signal'].iloc[i] != 0:
                signals.at[signals.index[i], 'a_position'] = signals['a_signal'].iloc[i]
            else:
                signals.at[signals.index[i], 'a_position'] = signals['a_position'].iloc[i-1]
                
            if signals['h_signal'].iloc[i] != 0:
                signals.at[signals.index[i], 'h_position'] = signals['h_signal'].iloc[i]
            else:
                signals.at[signals.index[i], 'h_position'] = signals['h_position'].iloc[i-1]
        
        return signals
    
    def backtest(self, data, initial_capital=100000.0):
        """
        Backtest the AH Premium strategy
        
        Parameters:
        data (DataFrame): DataFrame with A-share and H-share prices
        initial_capital (float): Starting capital for backtest
        
        Returns:
        DataFrame: Backtest results including portfolio values
        """
        # Generate signals
        signals = self.generate_signals(data)
        
        # Create portfolio dataframe
        portfolio = signals.copy()
        portfolio['a_holdings_shares'] = 0  # Number of A-shares held
        portfolio['h_holdings_shares'] = 0  # Number of H-shares held
        portfolio['a_holdings_value'] = 0.0  # Value of A-shares in CNY
        portfolio['h_holdings_value'] = 0.0  # Value of H-shares in CNY
        portfolio['cash_cny'] = initial_capital / 2  # Cash in CNY
        portfolio['cash_hkd'] = (initial_capital / 2) / portfolio['exchange_rate']  # Cash in HKD
        portfolio['total_cny'] = initial_capital  # Total value in CNY
        
        # Transaction costs
        a_commission_rate = 0.0005  # A-shares commission: 0.05%
        a_stamp_tax_rate = 0.001    # A-shares stamp tax: 0.1% for sell only
        h_commission_rate = 0.0007  # H-shares commission: 0.07%
        h_stamp_tax_rate = 0.001    # H-shares stamp tax: 0.1% for sell only
        
        for i in range(1, len(portfolio)):
            # Handle A-shares transactions
            if portfolio['a_position'].iloc[i] != portfolio['a_position'].iloc[i-1]:
                # A-shares price with T+1 rule (for selling)
                a_price = portfolio['a_price'].iloc[i]
                
                if portfolio['a_position'].iloc[i] > 0:  # Buy A-shares
                    # Calculate max shares buyable with cash (CNY)
                    a_shares_to_buy = int((portfolio['cash_cny'].iloc[i-1] * 0.95) // 
                                        (a_price * (1 + a_commission_rate)))
                    
                    # Buy 100 shares at a time (A-shares rule)
                    a_shares_to_buy = (a_shares_to_buy // 100) * 100
                    
                    if a_shares_to_buy > 0:
                        a_cost = a_shares_to_buy * a_price
                        a_commission = a_cost * a_commission_rate
                        portfolio.at[portfolio.index[i], 'a_holdings_shares'] = a_shares_to_buy
                        portfolio.at[portfolio.index[i], 'cash_cny'] = portfolio['cash_cny'].iloc[i-1] - a_cost - a_commission
                    else:
                        portfolio.at[portfolio.index[i], 'a_holdings_shares'] = 0
                        portfolio.at[portfolio.index[i], 'cash_cny'] = portfolio['cash_cny'].iloc[i-1]
                
                elif portfolio['a_position'].iloc[i] < 0:  # Sell A-shares
                    # For simplicity, assume we can short A-shares (not typical in real A-shares market)
                    a_shares_to_sell = int((portfolio['cash_cny'].iloc[i-1] * 0.95) // 
                                         (a_price * (1 + a_commission_rate + a_stamp_tax_rate)))
                    
                    # Sell 100 shares at a time
                    a_shares_to_sell = (a_shares_to_sell // 100) * 100
                    a_shares_to_sell = -a_shares_to_sell  # Negative for short position
                    
                    if a_shares_to_sell < 0:
                        a_revenue = -a_shares_to_sell * a_price
                        a_commission = a_revenue * a_commission_rate
                        a_stamp_tax = a_revenue * a_stamp_tax_rate
                        portfolio.at[portfolio.index[i], 'a_holdings_shares'] = a_shares_to_sell
                        portfolio.at[portfolio.index[i], 'cash_cny'] = portfolio['cash_cny'].iloc[i-1] + a_revenue - a_commission - a_stamp_tax
                    else:
                        portfolio.at[portfolio.index[i], 'a_holdings_shares'] = 0
                        portfolio.at[portfolio.index[i], 'cash_cny'] = portfolio['cash_cny'].iloc[i-1]
                
                else:  # No position
                    portfolio.at[portfolio.index[i], 'a_holdings_shares'] = 0
                    portfolio.at[portfolio.index[i], 'cash_cny'] = portfolio['cash_cny'].iloc[i-1]
            
            else:  # No change in position
                portfolio.at[portfolio.index[i], 'a_holdings_shares'] = portfolio['a_holdings_shares'].iloc[i-1]
                portfolio.at[portfolio.index[i], 'cash_cny'] = portfolio['cash_cny'].iloc[i-1]
            
            # Handle H-shares transactions
            if portfolio['h_position'].iloc[i] != portfolio['h_position'].iloc[i-1]:
                # H-shares price
                h_price = portfolio['h_price'].iloc[i]
                
                if portfolio['h_position'].iloc[i] > 0:  # Buy H-shares
                    # Calculate max shares buyable with cash (HKD)
                    h_shares_to_buy = int((portfolio['cash_hkd'].iloc[i-1] * 0.95) // 
                                        (h_price * (1 + h_commission_rate)))
                    
                    # Buy 500 shares at a time (H-shares typical lot)
                    h_shares_to_buy = (h_shares_to_buy // 500) * 500
                    
                    if h_shares_to_buy > 0:
                        h_cost = h_shares_to_buy * h_price
                        h_commission = h_cost * h_commission_rate
                        portfolio.at[portfolio.index[i], 'h_holdings_shares'] = h_shares_to_buy
                        portfolio.at[portfolio.index[i], 'cash_hkd'] = portfolio['cash_hkd'].iloc[i-1] - h_cost - h_commission
                    else:
                        portfolio.at[portfolio.index[i], 'h_holdings_shares'] = 0
                        portfolio.at[portfolio.index[i], 'cash_hkd'] = portfolio['cash_hkd'].iloc[i-1]
                
                elif portfolio['h_position'].iloc[i] < 0:  # Sell H-shares
                    # H-shares can be shorted more easily
                    h_shares_to_sell = int((portfolio['cash_hkd'].iloc[i-1] * 0.95) // 
                                         (h_price * (1 + h_commission_rate + h_stamp_tax_rate)))
                    
                    # Sell 500 shares at a time
                    h_shares_to_sell = (h_shares_to_sell // 500) * 500
                    h_shares_to_sell = -h_shares_to_sell  # Negative for short position
                    
                    if h_shares_to_sell < 0:
                        h_revenue = -h_shares_to_sell * h_price
                        h_commission = h_revenue * h_commission_rate
                        h_stamp_tax = h_revenue * h_stamp_tax_rate
                        portfolio.at[portfolio.index[i], 'h_holdings_shares'] = h_shares_to_sell
                        portfolio.at[portfolio.index[i], 'cash_hkd'] = portfolio['cash_hkd'].iloc[i-1] + h_revenue - h_commission - h_stamp_tax
                    else:
                        portfolio.at[portfolio.index[i], 'h_holdings_shares'] = 0
                        portfolio.at[portfolio.index[i], 'cash_hkd'] = portfolio['cash_hkd'].iloc[i-1]
                
                else:  # No position
                    portfolio.at[portfolio.index[i], 'h_holdings_shares'] = 0
                    portfolio.at[portfolio.index[i], 'cash_hkd'] = portfolio['cash_hkd'].iloc[i-1]
            
            else:  # No change in position
                portfolio.at[portfolio.index[i], 'h_holdings_shares'] = portfolio['h_holdings_shares'].iloc[i-1]
                portfolio.at[portfolio.index[i], 'cash_hkd'] = portfolio['cash_hkd'].iloc[i-1]
            
            # Calculate holdings value
            a_price = portfolio['a_price'].iloc[i]
            h_price = portfolio['h_price'].iloc[i]
            exchange_rate = portfolio['exchange_rate'].iloc[i]
            
            portfolio.at[portfolio.index[i], 'a_holdings_value'] = portfolio['a_holdings_shares'].iloc[i] * a_price
            h_value_hkd = portfolio['h_holdings_shares'].iloc[i] * h_price
            portfolio.at[portfolio.index[i], 'h_holdings_value'] = h_value_hkd * exchange_rate  # Convert to CNY
            
            # Calculate total portfolio value in CNY
            portfolio.at[portfolio.index[i], 'total_cny'] = (
                portfolio['a_holdings_value'].iloc[i] + 
                portfolio['h_holdings_value'].iloc[i] + 
                portfolio['cash_cny'].iloc[i] + 
                portfolio['cash_hkd'].iloc[i] * exchange_rate
            )
        
        # Calculate returns
        portfolio['returns'] = portfolio['total_cny'].pct_change()
        portfolio['strategy_returns'] = portfolio['total_cny'] / portfolio['total_cny'].iloc[0] - 1
        
        # Create a benchmark (simple average of A and H returns)
        portfolio['a_returns'] = portfolio['a_price'].pct_change()
        portfolio['h_returns_cny'] = (portfolio['h_price'] * portfolio['exchange_rate']).pct_change()
        portfolio['benchmark_returns'] = (portfolio['a_returns'] + portfolio['h_returns_cny']) / 2
        portfolio['benchmark_cumulative'] = (1 + portfolio['benchmark_returns']).cumprod() - 1
        
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
        plt.plot(results['total_cny'])
        plt.title('Portfolio Value (CNY)')
        plt.grid(True)
        
        # Plot strategy vs benchmark returns
        plt.subplot(312)
        plt.plot(results['strategy_returns'], label='Strategy')
        plt.plot(results['benchmark_cumulative'], label='Benchmark')
        plt.title('Strategy vs Benchmark Returns')
        plt.legend()
        plt.grid(True)
        
        # Plot AH premium and positions
        plt.subplot(313)
        plt.plot(results['ah_premium'], label='AH Premium (%)')
        plt.axhline(y=self.entry_threshold, color='r', linestyle='--', label='Entry Threshold')
        plt.axhline(y=-self.entry_threshold, color='r', linestyle='--')
        plt.axhline(y=self.exit_threshold, color='g', linestyle='--', label='Exit Threshold')
        plt.axhline(y=-self.exit_threshold, color='g', linestyle='--')
        
        # Plot buy/sell signals
        buy_a = results[(results['a_signal'] == 1) & (results['a_signal'].shift(1) != 1)]
        sell_a = results[(results['a_signal'] == -1) & (results['a_signal'].shift(1) != -1)]
        buy_h = results[(results['h_signal'] == 1) & (results['h_signal'].shift(1) != 1)]
        sell_h = results[(results['h_signal'] == -1) & (results['h_signal'].shift(1) != -1)]
        
        plt.scatter(buy_a.index, buy_a['ah_premium'], marker='^', c='g', label='Buy A', s=50)
        plt.scatter(sell_a.index, sell_a['ah_premium'], marker='v', c='r', label='Sell A', s=50)
        plt.scatter(buy_h.index, buy_h['ah_premium'], marker='o', c='g', label='Buy H', s=50)
        plt.scatter(sell_h.index, sell_h['ah_premium'], marker='o', c='r', label='Sell H', s=50)
        
        plt.title('AH Premium and Signals')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # For demonstration, create dummy data
    dates = pd.date_range(start='2021-01-01', end='2021-12-31')
    np.random.seed(42)
    
    # Create price data for a fictitious AH pair
    a_prices = np.random.normal(50, 1, len(dates)).cumsum() + 50
    h_prices = np.random.normal(45, 1, len(dates)).cumsum() + 45
    
    # Add mean-reverting element to create AH premium
    premium = np.zeros(len(dates))
    premium[0] = (a_prices[0] / (h_prices[0] * 1.1) - 1) * 100
    
    for i in range(1, len(dates)):
        # Add some mean-reversion to premium
        if premium[i-1] > 20:  # If premium is high
            a_prices[i] *= 0.99  # A-shares price decreases
            h_prices[i] *= 1.01  # H-shares price increases
        elif premium[i-1] < -20:  # If premium is low
            a_prices[i] *= 1.01  # A-shares price increases
            h_prices[i] *= 0.99  # H-shares price decreases
        
        # Calculate new premium
        premium[i] = (a_prices[i] / (h_prices[i] * 1.1) - 1) * 100
    
    # Exchange rate (CNY/HKD)
    exchange_rate = np.ones(len(dates)) * 0.9
    
    # Create test dataframe
    df = pd.DataFrame({
        'a_price': a_prices,
        'h_price': h_prices,
        'exchange_rate': exchange_rate
    }, index=dates)
    
    # Initialize strategy
    strategy = AHPremiumStrategy(entry_threshold=20, exit_threshold=5)
    
    # Backtest
    results = strategy.backtest(df)
    
    # Plot results
    # strategy.plot_results(results)
    
    # Print key metrics
    final_value = results['total_cny'].iloc[-1]
    initial_value = results['total_cny'].iloc[0]
    returns = (final_value / initial_value - 1) * 100
    print(f"Final Portfolio Value: {final_value:.2f} CNY")
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
# visualization.py
# Common visualization functions for quantitative trading strategies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta

# Set seaborn style
sns.set_style('whitegrid')


def plot_equity_curve(portfolio_values, benchmark_values=None, title='Strategy Performance', figsize=(12, 6)):
    """
    Plot equity curve of a strategy vs. benchmark
    
    Parameters:
    portfolio_values (Series): Series of portfolio values indexed by date
    benchmark_values (Series, optional): Series of benchmark values indexed by date
    title (str): Plot title
    figsize (tuple): Figure size
    
    Returns:
    matplotlib.figure.Figure: Plot figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot strategy equity curve
    portfolio_values.plot(ax=ax, label='Strategy', linewidth=2)
    
    # Plot benchmark if provided
    if benchmark_values is not None:
        benchmark_values.plot(ax=ax, label='Benchmark', linewidth=2, alpha=0.7)
    
    # Format plot
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Portfolio Value')
    ax.set_xlabel('Date')
    ax.legend()
    
    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig


def plot_returns_distribution(returns, benchmark_returns=None, title='Returns Distribution', figsize=(12, 6)):
    """
    Plot distribution of returns
    
    Parameters:
    returns (Series): Series of strategy returns
    benchmark_returns (Series, optional): Series of benchmark returns
    title (str): Plot title
    figsize (tuple): Figure size
    
    Returns:
    matplotlib.figure.Figure: Plot figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot strategy returns distribution
    sns.histplot(returns, bins=50, alpha=0.7, label='Strategy', ax=ax, kde=True)
    
    # Plot benchmark returns distribution if provided
    if benchmark_returns is not None:
        sns.histplot(benchmark_returns, bins=50, alpha=0.5, label='Benchmark', ax=ax, kde=True)
    
    # Add vertical lines for mean returns
    ax.axvline(returns.mean(), color='blue', linestyle='--', linewidth=2, label='Strategy Mean')
    if benchmark_returns is not None:
        ax.axvline(benchmark_returns.mean(), color='orange', linestyle='--', linewidth=2, label='Benchmark Mean')
    
    # Format plot
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Daily Returns')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_drawdowns(equity_curve, n_drawdowns=5, title='Drawdown Periods', figsize=(12, 6)):
    """
    Plot top drawdowns
    
    Parameters:
    equity_curve (Series): Series of portfolio values indexed by date
    n_drawdowns (int): Number of largest drawdowns to highlight
    title (str): Plot title
    figsize (tuple): Figure size
    
    Returns:
    matplotlib.figure.Figure: Plot figure
    """
    # Calculate drawdowns
    cum_returns = equity_curve / equity_curve.iloc[0]
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns / running_max - 1) * 100  # Convert to percentage
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot drawdowns
    drawdowns.plot(ax=ax, linewidth=2, color='red', alpha=0.7)
    
    # Find drawdown periods
    is_drawdown = drawdowns < 0
    drawdown_start = is_drawdown.astype(int).diff() == 1
    drawdown_end = is_drawdown.astype(int).diff() == -1
    
    start_dates = drawdowns.index[drawdown_start]
    end_dates = drawdowns.index[drawdown_end]
    
    # If we're still in a drawdown at the end of the period
    if len(start_dates) > len(end_dates):
        end_dates = end_dates.append(pd.Index([drawdowns.index[-1]]))
    
    # Create list of drawdown periods
    periods = []
    for i in range(len(start_dates)):
        period_drawdown = drawdowns.loc[start_dates[i]:end_dates[i]]
        max_drawdown = period_drawdown.min()
        max_drawdown_date = period_drawdown.idxmin()
        periods.append((start_dates[i], end_dates[i], max_drawdown_date, max_drawdown))
    
    # Sort by drawdown size
    periods.sort(key=lambda x: x[3])
    
    # Highlight top n drawdowns
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, min(n_drawdowns, len(periods))))
    
    for i, (start, end, max_date, max_dd) in enumerate(periods[:n_drawdowns]):
        ax.fill_between(drawdowns.loc[start:end].index, 0, drawdowns.loc[start:end], 
                         color=colors[i], alpha=0.3)
        ax.annotate(f'{max_dd:.1f}%', xy=(max_date, max_dd), 
                   xytext=(max_date + timedelta(days=5), max_dd * 0.8),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Format plot
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Drawdown (%)')
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig


def plot_monthly_returns_heatmap(returns, title='Monthly Returns (%)', figsize=(12, 8)):
    """
    Plot monthly returns heatmap
    
    Parameters:
    returns (Series): Series of daily returns indexed by date
    title (str): Plot title
    figsize (tuple): Figure size
    
    Returns:
    matplotlib.figure.Figure: Plot figure
    """
    # Resample returns to monthly
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
    
    # Create a pivot table of monthly returns by year and month
    returns_table = pd.DataFrame([])
    
    for year in range(returns.index.year.min(), returns.index.year.max() + 1):
        for month in range(1, 13):
            month_name = datetime(year, month, 1).strftime('%b')
            try:
                # Try to get the return for this month
                ret = monthly_returns.loc[f"{year}-{month:02d}"].item()
            except:
                # If the month is not in the data, use NaN
                ret = np.nan
            
            # Add to the table
            returns_table.loc[month_name, year] = ret
    
    # Reorder rows to have Jan-Dec
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    returns_table = returns_table.reindex(month_order)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(returns_table, annot=True, fmt='.2f', center=0, cmap='RdYlGn',
               linewidths=0.5, ax=ax, cbar_kws={'label': 'Return (%)'})
    
    # Calculate annual returns
    annual_returns = returns.resample('A').apply(lambda x: (1 + x).prod() - 1) * 100
    annual_return_values = [f"{annual_returns.loc[str(year)].item():.2f}%" 
                            if str(year) in annual_returns.index 
                            else "N/A" 
                            for year in returns_table.columns]
    
    # Add annual returns at the bottom
    ax.set_title(title, fontsize=14)
    plt.figtext(0.1, 0.01, f"Annual Returns: " + " | ".join([f"{year}: {ret}" 
                                                             for year, ret 
                                                             in zip(returns_table.columns, 
                                                                   annual_return_values)]),
               fontsize=10)
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.1)  # Add space for the annual returns text
    return fig


def plot_rolling_metrics(returns, benchmark_returns=None, window=252, title='Rolling Performance Metrics', figsize=(14, 10)):
    """
    Plot rolling performance metrics
    
    Parameters:
    returns (Series): Series of strategy returns
    benchmark_returns (Series, optional): Series of benchmark returns
    window (int): Rolling window size in days
    title (str): Plot title
    figsize (tuple): Figure size
    
    Returns:
    matplotlib.figure.Figure: Plot figure
    """
    # Ensure we have enough data
    if len(returns) < window:
        raise ValueError(f"Not enough data for rolling window of size {window}")
    
    # Calculate rolling returns (annualized)
    rolling_returns = returns.rolling(window=window).apply(lambda x: (1 + x).prod() - 1) * (252 / window)
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
    rolling_sharpe = rolling_returns / rolling_vol
    
    # Calculate rolling drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_dd = pd.Series(index=returns.index, dtype='float64')
    
    for i in range(window, len(returns) + 1):
        window_cum_returns = cum_returns.iloc[i-window:i]
        rolling_dd.iloc[i-1] = (window_cum_returns / window_cum_returns.cummax() - 1).min()
    
    # Calculate rolling beta and alpha if benchmark is provided
    if benchmark_returns is not None:
        rolling_beta = pd.Series(index=returns.index, dtype='float64')
        rolling_alpha = pd.Series(index=returns.index, dtype='float64')
        
        for i in range(window, len(returns) + 1):
            window_returns = returns.iloc[i-window:i]
            window_benchmark = benchmark_returns.iloc[i-window:i]
            
            cov = np.cov(window_returns, window_benchmark)[0, 1]
            var = np.var(window_benchmark)
            beta = cov / var
            
            rolling_beta.iloc[i-1] = beta
            
            # Calculate alpha (annualized)
            r_f = 0.0  # Assume risk-free rate is 0 for simplicity
            alpha = window_returns.mean() * 252 - r_f - beta * (window_benchmark.mean() * 252 - r_f)
            rolling_alpha.iloc[i-1] = alpha
    
    # Create figure and plot
    fig, axs = plt.subplots(4 if benchmark_returns is None else 5, 1, figsize=figsize, sharex=True)
    
    # Set title
    fig.suptitle(title, fontsize=16)
    
    # Plot rolling metrics
    axs[0].plot(rolling_returns, linewidth=2)
    axs[0].set_title('Rolling Annualized Return')
    axs[0].set_ylabel('Return')
    
    axs[1].plot(rolling_vol, linewidth=2, color='orange')
    axs[1].set_title('Rolling Annualized Volatility')
    axs[1].set_ylabel('Volatility')
    
    axs[2].plot(rolling_sharpe, linewidth=2, color='green')
    axs[2].set_title('Rolling Sharpe Ratio')
    axs[2].set_ylabel('Sharpe')
    
    axs[3].plot(rolling_dd, linewidth=2, color='red')
    axs[3].set_title('Rolling Maximum Drawdown')
    axs[3].set_ylabel('Drawdown')
    
    if benchmark_returns is not None:
        axs[4].plot(rolling_beta, linewidth=2, color='purple')
        axs[4].plot(rolling_alpha, linewidth=2, color='blue', linestyle='--')
        axs[4].set_title('Rolling Beta and Alpha')
        axs[4].set_ylabel('Beta / Alpha')
        axs[4].legend(['Beta', 'Alpha'])
    
    # Format x-axis
    if isinstance(returns.index, pd.DatetimeIndex):
        for ax in axs:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)  # Adjust for main title
    
    return fig


def plot_positions_over_time(positions_df, prices_df=None, title='Positions Over Time', figsize=(12, 6)):
    """
    Plot positions over time
    
    Parameters:
    positions_df (DataFrame): DataFrame with positions (index=dates, columns=symbols)
    prices_df (DataFrame, optional): DataFrame with prices for position valuation
    title (str): Plot title
    figsize (tuple): Figure size
    
    Returns:
    matplotlib.figure.Figure: Plot figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # If prices are provided, use them to calculate position values
    if prices_df is not None:
        # Calculate position values
        position_values = positions_df.multiply(prices_df)
        position_values.plot(ax=ax, linewidth=2, alpha=0.7)
        ylabel = 'Position Value'
    else:
        # Just plot the number of shares/contracts
        positions_df.plot(ax=ax, linewidth=2, alpha=0.7)
        ylabel = 'Number of Shares/Contracts'
    
    # Format plot
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Date')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Format x-axis dates
    if isinstance(positions_df.index, pd.DatetimeIndex):
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def plot_trade_analysis(trades_df, title='Trade Analysis', figsize=(14, 10)):
    """
    Plot trade analysis visualizations
    
    Parameters:
    trades_df (DataFrame): DataFrame with trade details (entry/exit dates, prices, P&L)
    title (str): Plot title
    figsize (tuple): Figure size
    
    Returns:
    matplotlib.figure.Figure: Plot figure
    """
    # Check if we have any trades
    if len(trades_df) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No trades to analyze', horizontalalignment='center',
               verticalalignment='center', fontsize=14)
        return fig
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Calculate trade metrics
    trades_df['duration'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
    trades_df['return'] = trades_df['pnl'] / (trades_df['entry_price'] * trades_df['quantity'])
    
    # 1. Plot P&L distribution
    sns.histplot(trades_df['pnl'], bins=20, kde=True, ax=axs[0, 0])
    axs[0, 0].axvline(0, color='red', linestyle='--')
    axs[0, 0].set_title('P&L Distribution')
    axs[0, 0].set_xlabel('P&L')
    
    # 2. Plot P&L by duration
    axs[0, 1].scatter(trades_df['duration'], trades_df['pnl'], alpha=0.7)
    axs[0, 1].axhline(0, color='red', linestyle='--')
    axs[0, 1].set_title('P&L by Trade Duration')
    axs[0, 1].set_xlabel('Duration (days)')
    axs[0, 1].set_ylabel('P&L')
    
    # 3. Plot cumulative P&L
    cum_pnl = trades_df['pnl'].cumsum()
    axs[1, 0].plot(cum_pnl.values)
    axs[1, 0].set_title('Cumulative P&L')
    axs[1, 0].set_xlabel('Trade Number')
    axs[1, 0].set_ylabel('Cumulative P&L')
    
    # 4. Plot P&L by symbol (top symbols)
    symbol_pnl = trades_df.groupby('symbol')['pnl'].sum().sort_values()
    top_n = min(10, len(symbol_pnl))
    symbol_pnl.tail(top_n).plot(kind='barh', ax=axs[1, 1])
    axs[1, 1].set_title(f'P&L by Symbol (Top {top_n})')
    axs[1, 1].set_xlabel('P&L')
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)  # Adjust for main title
    
    return fig


def create_performance_report(returns, benchmark_returns=None, title='Strategy Performance Report'):
    """
    Create a comprehensive performance report with multiple plots
    
    Parameters:
    returns (Series): Series of strategy returns
    benchmark_returns (Series, optional): Series of benchmark returns
    title (str): Report title
    
    Returns:
    list: List of matplotlib figures
    """
    # Calculate equity curves
    equity_curve = (1 + returns).cumprod()
    benchmark_equity = (1 + benchmark_returns).cumprod() if benchmark_returns is not None else None
    
    # Create figures
    figures = []
    
    # 1. Equity curve
    figures.append(plot_equity_curve(equity_curve, benchmark_equity, title='Strategy vs Benchmark'))
    
    # 2. Drawdowns
    figures.append(plot_drawdowns(equity_curve, title='Strategy Drawdowns'))
    
    # 3. Returns distribution
    figures.append(plot_returns_distribution(returns, benchmark_returns, title='Returns Distribution'))
    
    # 4. Monthly returns heatmap
    figures.append(plot_monthly_returns_heatmap(returns, title='Monthly Returns (%)'))
    
    # 5. Rolling metrics
    window_size = min(252, len(returns) // 2)  # Use smaller window if not enough data
    if window_size >= 20:  # Only include if we have enough data
        figures.append(plot_rolling_metrics(returns, benchmark_returns, window=window_size, title='Rolling Metrics'))
    
    return figures


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2021-12-31')
    
    # Create returns
    returns = pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates)
    benchmark_returns = pd.Series(np.random.normal(0.0004, 0.012, len(dates)), index=dates)
    
    # Calculate equity curves
    equity_curve = (1 + returns).cumprod() * 100000
    benchmark_equity = (1 + benchmark_returns).cumprod() * 100000
    
    # Plot equity curve
    fig1 = plot_equity_curve(equity_curve, benchmark_equity, title='Strategy vs Benchmark')
    
    # Plot returns distribution
    fig2 = plot_returns_distribution(returns, benchmark_returns, title='Returns Distribution')
    
    # Plot drawdowns
    fig3 = plot_drawdowns(equity_curve, title='Strategy Drawdowns')
    
    # Plot monthly returns heatmap
    fig4 = plot_monthly_returns_heatmap(returns, title='Monthly Returns (%)')
    
    # Plot rolling metrics
    fig5 = plot_rolling_metrics(returns, benchmark_returns, window=60, title='Rolling Metrics')
    
    # Show plots
    plt.show()
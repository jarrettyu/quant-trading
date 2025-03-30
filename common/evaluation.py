# evaluation.py
# Common evaluation functions for quantitative trading strategies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta

class StrategyEvaluator:
    """
    Class for evaluating quantitative trading strategies
    """
    
    def __init__(self, returns, benchmark_returns=None, risk_free_rate=0.0):
        """
        Initialize the evaluator with strategy returns
        
        Parameters:
        returns (Series): Strategy returns (daily)
        benchmark_returns (Series, optional): Benchmark returns (daily)
        risk_free_rate (float): Risk-free rate (annualized)
        """
        # Validate inputs
        if not isinstance(returns, pd.Series):
            if isinstance(returns, pd.DataFrame) and len(returns.columns) == 1:
                returns = returns.iloc[:, 0]
            else:
                returns = pd.Series(returns)
        
        if benchmark_returns is not None and not isinstance(benchmark_returns, pd.Series):
            if isinstance(benchmark_returns, pd.DataFrame) and len(benchmark_returns.columns) == 1:
                benchmark_returns = benchmark_returns.iloc[:, 0]
            else:
                benchmark_returns = pd.Series(benchmark_returns)
        
        # Store inputs
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        
        # Calculate daily risk-free rate
        self.daily_risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1
    
    def calc_cumulative_returns(self):
        """
        Calculate cumulative returns
        
        Returns:
        Series: Cumulative returns
        """
        cumulative_returns = (1 + self.returns).cumprod() - 1
        return cumulative_returns
    
    def calc_annual_return(self):
        """
        Calculate annualized return
        
        Returns:
        float: Annualized return
        """
        # Calculate cumulative return
        cumulative_return = (1 + self.returns).prod() - 1
        
        # Calculate number of years
        num_days = len(self.returns)
        num_years = num_days / 252
        
        # Calculate annualized return
        annual_return = (1 + cumulative_return) ** (1 / num_years) - 1
        
        return annual_return
    
    def calc_volatility(self, annualized=True):
        """
        Calculate return volatility
        
        Parameters:
        annualized (bool): Whether to annualize volatility
        
        Returns:
        float: Return volatility
        """
        # Calculate daily volatility
        daily_volatility = self.returns.std()
        
        # Annualize if requested
        if annualized:
            volatility = daily_volatility * np.sqrt(252)
        else:
            volatility = daily_volatility
        
        return volatility
    
    def calc_sharpe_ratio(self):
        """
        Calculate Sharpe ratio
        
        Returns:
        float: Sharpe ratio
        """
        # Calculate excess returns
        excess_returns = self.returns - self.daily_risk_free_rate
        
        # Calculate Sharpe ratio
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        return sharpe_ratio
    
    def calc_sortino_ratio(self):
        """
        Calculate Sortino ratio
        
        Returns:
        float: Sortino ratio
        """
        # Calculate excess returns
        excess_returns = self.returns - self.daily_risk_free_rate
        
        # Calculate downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        
        # Handle case where there are no negative returns
        if downside_deviation == 0 or np.isnan(downside_deviation):
            return np.nan
        
        # Calculate Sortino ratio
        sortino_ratio = excess_returns.mean() * 252 / downside_deviation
        
        return sortino_ratio
    
    def calc_max_drawdown(self):
        """
        Calculate maximum drawdown
        
        Returns:
        tuple: (max_drawdown, start_date, end_date, recovery_date)
        """
        # Calculate cumulative returns
        cumulative_returns = self.calc_cumulative_returns()
        
        # Calculate running maximum
        running_max = cumulative_returns.cummax()
        
        # Calculate drawdown
        drawdown = (cumulative_returns / running_max - 1)
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        
        # Find drawdown dates if returns have a datetime index
        if isinstance(self.returns.index, pd.DatetimeIndex):
            # Find drawdown start date (last peak before worst drawdown)
            drawdown_end = drawdown.idxmin()
            temp = cumulative_returns.loc[:drawdown_end]
            drawdown_start = temp.idxmax()
            
            # Find recovery date (first time cumulative returns exceeds previous peak)
            if drawdown_end < cumulative_returns.index[-1]:
                temp = cumulative_returns.loc[drawdown_end:]
                temp_peak = cumulative_returns.loc[drawdown_start]
                recovery_date = temp[temp >= temp_peak].index[0] if any(temp >= temp_peak) else None
            else:
                recovery_date = None
            
            return (max_drawdown, drawdown_start, drawdown_end, recovery_date)
        else:
            return (max_drawdown, None, None, None)
    
    def calc_calmar_ratio(self):
        """
        Calculate Calmar ratio (annualized return / maximum drawdown)
        
        Returns:
        float: Calmar ratio
        """
        # Calculate annualized return
        annual_return = self.calc_annual_return()
        
        # Calculate maximum drawdown
        max_drawdown = self.calc_max_drawdown()[0]
        
        # Handle case where there is no drawdown
        if max_drawdown == 0:
            return np.inf
        
        # Calculate Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown)
        
        return calmar_ratio
    
    def calc_omega_ratio(self, threshold=0.0):
        """
        Calculate Omega ratio
        
        Parameters:
        threshold (float): Return threshold
        
        Returns:
        float: Omega ratio
        """
        # Calculate returns above and below threshold
        returns_above = self.returns[self.returns > threshold] - threshold
        returns_below = threshold - self.returns[self.returns < threshold]
        
        # Handle case where there are no returns below threshold
        if len(returns_below) == 0:
            return np.inf
        
        # Calculate Omega ratio
        omega_ratio = returns_above.sum() / returns_below.sum()
        
        return omega_ratio
    
    def calc_var(self, percentile=5):
        """
        Calculate Value at Risk (VaR)
        
        Parameters:
        percentile (float): Percentile for VaR
        
        Returns:
        float: VaR
        """
        # Calculate VaR
        var = np.percentile(self.returns, percentile)
        
        return var
    
    def calc_cvar(self, percentile=5):
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
        
        Parameters:
        percentile (float): Percentile for CVaR
        
        Returns:
        float: CVaR
        """
        # Calculate VaR
        var = self.calc_var(percentile)
        
        # Calculate CVaR (average of returns below VaR)
        cvar = self.returns[self.returns <= var].mean()
        
        return cvar
    
    def calc_beta(self):
        """
        Calculate beta relative to benchmark
        
        Returns:
        float: Beta
        """
        if self.benchmark_returns is None:
            return None
        
        # Calculate covariance between strategy and benchmark returns
        cov = np.cov(self.returns, self.benchmark_returns)[0, 1]
        
        # Calculate benchmark variance
        benchmark_var = self.benchmark_returns.var()
        
        # Calculate beta
        beta = cov / benchmark_var
        
        return beta
    
    def calc_alpha(self):
        """
        Calculate alpha (Jensen's alpha)
        
        Returns:
        float: Alpha (annualized)
        """
        if self.benchmark_returns is None:
            return None
        
        # Calculate beta
        beta = self.calc_beta()
        
        # Calculate average returns
        avg_return = self.returns.mean()
        avg_benchmark_return = self.benchmark_returns.mean()
        
        # Calculate alpha (daily)
        alpha = avg_return - (self.daily_risk_free_rate + beta * (avg_benchmark_return - self.daily_risk_free_rate))
        
        # Annualize alpha
        alpha_annualized = alpha * 252
        
        return alpha_annualized
    
    def calc_information_ratio(self):
        """
        Calculate information ratio
        
        Returns:
        float: Information ratio
        """
        if self.benchmark_returns is None:
            return None
        
        # Calculate tracking error
        tracking_error = (self.returns - self.benchmark_returns).std() * np.sqrt(252)
        
        # Calculate excess return
        excess_return = self.calc_annual_return() - (
            (1 + self.benchmark_returns).prod() - 1
        ) ** (252 / len(self.benchmark_returns)) - 1
        
        # Calculate information ratio
        information_ratio = excess_return / tracking_error
        
        return information_ratio
    
    def calc_win_rate(self):
        """
        Calculate win rate (percentage of positive returns)
        
        Returns:
        float: Win rate
        """
        # Calculate win rate
        win_rate = (self.returns > 0).mean()
        
        return win_rate
    
    def calc_profit_factor(self):
        """
        Calculate profit factor (gross profit / gross loss)
        
        Returns:
        float: Profit factor
        """
        # Calculate gross profit and loss
        gross_profit = self.returns[self.returns > 0].sum()
        gross_loss = abs(self.returns[self.returns < 0].sum())
        
        # Handle case where there are no losses
        if gross_loss == 0:
            return np.inf
        
        # Calculate profit factor
        profit_factor = gross_profit / gross_loss
        
        return profit_factor
    
    def calc_average_return(self, positive=None):
        """
        Calculate average return
        
        Parameters:
        positive (bool, optional): If True, only positive returns; if False, only negative returns
        
        Returns:
        float: Average return
        """
        if positive is None:
            # All returns
            avg_return = self.returns.mean()
        elif positive:
            # Only positive returns
            avg_return = self.returns[self.returns > 0].mean()
        else:
            # Only negative returns
            avg_return = self.returns[self.returns < 0].mean()
        
        return avg_return
    
    def calc_skewness(self):
        """
        Calculate return skewness
        
        Returns:
        float: Skewness
        """
        # Calculate skewness
        skewness = stats.skew(self.returns)
        
        return skewness
    
    def calc_kurtosis(self):
        """
        Calculate return kurtosis
        
        Returns:
        float: Kurtosis
        """
        # Calculate kurtosis
        kurtosis = stats.kurtosis(self.returns)
        
        return kurtosis
    
    def calc_monthly_returns(self):
        """
        Calculate monthly returns
        
        Returns:
        DataFrame: Monthly returns
        """
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            return None
        
        # Resample returns to monthly frequency
        monthly_returns = self.returns.resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        return monthly_returns
    
    def calc_annual_returns(self):
        """
        Calculate annual returns
        
        Returns:
        DataFrame: Annual returns
        """
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            return None
        
        # Resample returns to annual frequency
        annual_returns = self.returns.resample('A').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        return annual_returns
    
    def get_summary_stats(self):
        """
        Get summary statistics for the strategy
        
        Returns:
        dict: Dictionary of summary statistics
        """
        # Calculate various metrics
        annual_return = self.calc_annual_return()
        volatility = self.calc_volatility()
        sharpe_ratio = self.calc_sharpe_ratio()
        sortino_ratio = self.calc_sortino_ratio()
        max_drawdown = self.calc_max_drawdown()[0]
        calmar_ratio = self.calc_calmar_ratio()
        omega_ratio = self.calc_omega_ratio()
        var = self.calc_var()
        cvar = self.calc_cvar()
        win_rate = self.calc_win_rate()
        profit_factor = self.calc_profit_factor()
        avg_return = self.calc_average_return()
        avg_pos_return = self.calc_average_return(positive=True)
        avg_neg_return = self.calc_average_return(positive=False)
        skewness = self.calc_skewness()
        kurtosis = self.calc_kurtosis()
        
        # Calculate benchmark-relative metrics if benchmark is available
        if self.benchmark_returns is not None:
            beta = self.calc_beta()
            alpha = self.calc_alpha()
            information_ratio = self.calc_information_ratio()
        else:
            beta = None
            alpha = None
            information_ratio = None
        
        # Create summary dictionary
        summary = {
            'Annual Return': annual_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown,
            'Calmar Ratio': calmar_ratio,
            'Omega Ratio': omega_ratio,
            'Value at Risk (5%)': var,
            'Conditional VaR (5%)': cvar,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'Average Daily Return': avg_return,
            'Average Positive Return': avg_pos_return,
            'Average Negative Return': avg_neg_return,
            'Return Skewness': skewness,
            'Return Kurtosis': kurtosis,
            'Beta': beta,
            'Alpha': alpha,
            'Information Ratio': information_ratio
        }
        
        return summary
    
    def plot_returns(self, figsize=(12, 6)):
        """
        Plot cumulative returns
        
        Parameters:
        figsize (tuple): Figure size
        
        Returns:
        matplotlib.figure.Figure: Plot figure
        """
        # Calculate cumulative returns
        cumulative_returns = self.calc_cumulative_returns()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot strategy returns
        ax.plot(cumulative_returns, label='Strategy')
        
        # Plot benchmark returns if available
        if self.benchmark_returns is not None:
            benchmark_cumulative_returns = (1 + self.benchmark_returns).cumprod() - 1
            ax.plot(benchmark_cumulative_returns, label='Benchmark')
        
        # Add labels and legend
        ax.set_title('Cumulative Returns')
        ax.set_ylabel('Return')
        ax.grid(True)
        ax.legend()
        
        return fig
    
    def plot_drawdowns(self, top_n=5, figsize=(12, 6)):
        """
        Plot top drawdowns
        
        Parameters:
        top_n (int): Number of top drawdowns to plot
        figsize (tuple): Figure size
        
        Returns:
        matplotlib.figure.Figure: Plot figure
        """
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            raise ValueError("Returns must have a DatetimeIndex for drawdown plotting")
        
        # Calculate cumulative returns
        cumulative_returns = self.calc_cumulative_returns()
        
        # Calculate running maximum
        running_max = cumulative_returns.cummax()
        
        # Calculate drawdown
        drawdown = (cumulative_returns / running_max - 1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot drawdowns
        ax.plot(drawdown)
        
        # Identify and highlight top drawdowns
        underwater = drawdown < 0
        underwater_periods = underwater.astype(int).diff()
        start_dates = underwater_periods[underwater_periods == 1].index
        end_dates = underwater_periods[underwater_periods == -1].index
        
        # Handle case where strategy is still underwater at end
        if len(start_dates) > len(end_dates):
            end_dates = end_dates.append(pd.DatetimeIndex([drawdown.index[-1]]))
        
        # Create list of drawdown periods
        drawdown_periods = []
        for i in range(len(start_dates)):
            period_drawdown = drawdown.loc[start_dates[i]:end_dates[i]]
            max_drawdown = period_drawdown.min()
            drawdown_periods.append((start_dates[i], end_dates[i], max_drawdown))
        
        # Sort drawdown periods by drawdown size
        drawdown_periods.sort(key=lambda x: x[2])
        
        # Highlight top N drawdowns
        colors = plt.cm.Reds(np.linspace(0.5, 0.9, top_n))
        for i, (start, end, max_dd) in enumerate(drawdown_periods[:top_n]):
            ax.fill_between(drawdown.loc[start:end].index, 0, drawdown.loc[start:end],
                          color=colors[i], alpha=0.3)
            ax.text(end, max_dd, f'{max_dd:.2%}', fontsize=10)
        
        # Add labels
        ax.set_title('Strategy Drawdowns')
        ax.set_ylabel('Drawdown')
        ax.grid(True)
        
        return fig
    
    def plot_monthly_returns_heatmap(self, figsize=(12, 8)):
        """
        Plot monthly returns heatmap
        
        Parameters:
        figsize (tuple): Figure size
        
        Returns:
        matplotlib.figure.Figure: Plot figure
        """
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            raise ValueError("Returns must have a DatetimeIndex for monthly heatmap")
        
        # Calculate monthly returns
        monthly_returns = self.calc_monthly_returns()
        
        # Convert to percentage
        monthly_returns_pct = monthly_returns * 100
        
        # Create DataFrame with month as rows and year as columns
        returns_table = pd.DataFrame(
            [
                [
                    monthly_returns_pct.loc[f"{y}-{m:02d}-01"].item() 
                    if f"{y}-{m:02d}-01" in monthly_returns_pct.index 
                    else np.nan 
                    for y in range(monthly_returns_pct.index.year.min(), monthly_returns_pct.index.year.max() + 1)
                ] 
                for m in range(1, 13)
            ],
            index=[
                'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
            ],
            columns=range(monthly_returns_pct.index.year.min(), monthly_returns_pct.index.year.max() + 1)
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(returns_table, annot=True, cmap='RdYlGn', fmt='.2f', center=0, ax=ax)
        
        # Add labels
        ax.set_title('Monthly Returns (%)')
        
        return fig
    
    def plot_return_distribution(self, bins=50, figsize=(12, 6)):
        """
        Plot return distribution
        
        Parameters:
        bins (int): Number of bins for histogram
        figsize (tuple): Figure size
        
        Returns:
        matplotlib.figure.Figure: Plot figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot return distribution
        sns.histplot(self.returns, bins=bins, kde=True, ax=ax)
        
        # Add labels
        ax.set_title('Return Distribution')
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        
        # Add vertical lines for mean and median
        ax.axvline(self.returns.mean(), color='r', linestyle='--', label='Mean')
        ax.axvline(self.returns.median(), color='g', linestyle='--', label='Median')
        
        # Add normal distribution
        if len(self.returns) > 30:
            x = np.linspace(self.returns.min(), self.returns.max(), 1000)
            y = stats.norm.pdf(x, self.returns.mean(), self.returns.std())
            ax.plot(x, y * len(self.returns) * (self.returns.max() - self.returns.min()) / bins,
                   color='k', linestyle='-', label='Normal')
        
        ax.legend()
        
        return fig
    
    def plot_rolling_metrics(self, window=252, figsize=(12, 12)):
        """
        Plot rolling metrics
        
        Parameters:
        window (int): Rolling window size
        figsize (tuple): Figure size
        
        Returns:
        matplotlib.figure.Figure: Plot figure
        """
        if len(self.returns) < window:
            raise ValueError(f"Not enough data for rolling window of size {window}")
        
        # Create figure
        fig, axs = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Calculate rolling metrics
        rolling_returns = self.returns.rolling(window=window).apply(
            lambda x: (1 + x).prod() - 1
        ) * (252 / window)  # Annualize
        
        rolling_vol = self.returns.rolling(window=window).std() * np.sqrt(252)
        
        rolling_sharpe = rolling_returns / rolling_vol
        
        # Calculate rolling drawdown
        rolling_dd = pd.Series(index=self.returns.index)
        for i in range(window, len(self.returns) + 1):
            temp_returns = self.returns.iloc[i-window:i]
            cum_returns = (1 + temp_returns).cumprod()
            max_dd = (cum_returns / cum_returns.cummax() - 1).min()
            rolling_dd.iloc[i-1] = max_dd
        
        # Plot rolling metrics
        axs[0].plot(rolling_returns)
        axs[0].set_title('Rolling Annualized Return')
        axs[0].grid(True)
        
        axs[1].plot(rolling_vol)
        axs[1].set_title('Rolling Annualized Volatility')
        axs[1].grid(True)
        
        axs[2].plot(rolling_sharpe)
        axs[2].set_title('Rolling Sharpe Ratio')
        axs[2].grid(True)
        
        axs[3].plot(rolling_dd)
        axs[3].set_title('Rolling Maximum Drawdown')
        axs[3].grid(True)
        
        plt.tight_layout()
        
        return fig


# Example usage
if __name__ == "__main__":
    # Create random returns for demo
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2021-12-31')
    returns = pd.Series(np.random.normal(0.0005, 0.01, len(dates)), index=dates)
    benchmark_returns = pd.Series(np.random.normal(0.0004, 0.012, len(dates)), index=dates)
    
    # Create evaluator
    evaluator = StrategyEvaluator(returns, benchmark_returns, risk_free_rate=0.02)
    
    # Print summary statistics
    summary = evaluator.get_summary_stats()
    for metric, value in summary.items():
        if value is not None:
            print(f"{metric}: {value:.6f}")
    
    # Plot cumulative returns
    evaluator.plot_returns()
    plt.show()
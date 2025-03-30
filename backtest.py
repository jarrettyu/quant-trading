#!/usr/bin/env python
# Main backtest script for running quantitative trading strategies

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import importlib.util
import json
import warnings
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from different markets
try:
    from a_shares.backtest.config import ASharesBacktestConfig
except ImportError:
    print("Warning: A-shares backtest config not found")

try:
    from hk_stocks.backtest.config import HKBacktestConfig
except ImportError:
    print("Warning: Hong Kong backtest config not found")

try:
    from us_market.backtest.config import USMarketBacktestConfig
except ImportError:
    print("Warning: US market backtest config not found")

# Import common modules
from common.evaluation import StrategyEvaluator
from common.visualization import create_performance_report
from common.utils import (
    save_backtest_results, 
    load_backtest_results, 
    ensure_datetime_index, 
    convert_price_format
)


def load_strategy(strategy_path):
    """
    Load strategy from file
    
    Parameters:
    strategy_path (str): Path to strategy file
    
    Returns:
    object: Strategy object
    """
    if not os.path.exists(strategy_path):
        raise FileNotFoundError(f"Strategy file not found: {strategy_path}")
    
    try:
        # Get the module name from the file path
        module_name = os.path.splitext(os.path.basename(strategy_path))[0]
        
        # Import the module
        spec = importlib.util.spec_from_file_location(module_name, strategy_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find the strategy class
        strategy_class = None
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and "Strategy" in name and name != "Strategy":
                strategy_class = obj
                break
        
        if strategy_class is None:
            raise ValueError(f"No strategy class found in {strategy_path}")
        
        # Create strategy instance
        return strategy_class()
    
    except Exception as e:
        print(f"Error loading strategy: {e}")
        return None


def load_config(market, config_path=None):
    """
    Load backtest configuration
    
    Parameters:
    market (str): Market type ('a_shares', 'hk_stocks', 'us_market')
    config_path (str, optional): Path to config file
    
    Returns:
    object: Configuration object
    """
    if config_path is not None and os.path.exists(config_path):
        try:
            # Load config from file
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Create config object based on market
            if market == 'a_shares':
                return ASharesBacktestConfig(**config_data)
            elif market == 'hk_stocks':
                return HKBacktestConfig(**config_data)
            elif market == 'us_market':
                return USMarketBacktestConfig(**config_data)
            else:
                raise ValueError(f"Unknown market: {market}")
        
        except Exception as e:
            print(f"Error loading config from file: {e}")
            print("Using default configuration instead")
    
    # Create default config
    try:
        if market == 'a_shares':
            return ASharesBacktestConfig()
        elif market == 'hk_stocks':
            return HKBacktestConfig()
        elif market == 'us_market':
            return USMarketBacktestConfig()
        else:
            raise ValueError(f"Unknown market: {market}")
    
    except Exception as e:
        print(f"Error creating default config: {e}")
        return None


def load_data(data_path, market):
    """
    Load price data from file
    
    Parameters:
    data_path (str): Path to data file
    market (str): Market type ('a_shares', 'hk_stocks', 'us_market')
    
    Returns:
    DataFrame: Price data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        # Get file extension
        ext = os.path.splitext(data_path)[1].lower()
        
        # Load data based on extension
        if ext == '.csv':
            data = pd.read_csv(data_path)
        elif ext == '.parquet':
            data = pd.read_parquet(data_path)
        elif ext == '.pickle':
            data = pd.read_pickle(data_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        # Convert to OHLC format
        data = convert_price_format(data, 'ohlc')
        
        # Ensure DatetimeIndex
        data = ensure_datetime_index(data)
        
        return data
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def run_backtest(strategy, data, config, benchmark_data=None):
    """
    Run backtest for a strategy
    
    Parameters:
    strategy (object): Strategy object
    data (DataFrame): Price data
    config (object): Configuration object
    benchmark_data (DataFrame, optional): Benchmark price data
    
    Returns:
    dict: Backtest results
    """
    try:
        # Initialize backtest parameters
        initial_capital = config.initial_capital
        start_date = pd.to_datetime(config.start_date)
        end_date = pd.to_datetime(config.end_date)
        
        # Filter data by date range
        if isinstance(data.index, pd.DatetimeIndex):
            data = data[(data.index >= start_date) & (data.index <= end_date)]
        else:
            print("Warning: Data index is not DatetimeIndex. Date filtering skipped.")
        
        # Filter benchmark data by date range
        if benchmark_data is not None and isinstance(benchmark_data.index, pd.DatetimeIndex):
            benchmark_data = benchmark_data[(benchmark_data.index >= start_date) & 
                                           (benchmark_data.index <= end_date)]
        
        # Generate signals
        print("Generating signals...")
        signals = strategy.generate_signals(data)
        
        # Run backtest
        print("Running backtest...")
        backtest_results = strategy.backtest(data, initial_capital=initial_capital)
        
        # Prepare return data
        if 'returns' in backtest_results:
            returns = backtest_results['returns']
        else:
            # Calculate returns from portfolio value
            portfolio_values = backtest_results.get('portfolio_value', backtest_results.get('total', None))
            if portfolio_values is not None:
                returns = portfolio_values.pct_change().fillna(0)
            else:
                print("Warning: Cannot calculate returns from backtest results")
                returns = pd.Series(index=data.index)
        
        # Prepare benchmark returns
        if benchmark_data is not None:
            benchmark_returns = benchmark_data['close'].pct_change().fillna(0)
            # Align benchmark returns with strategy returns
            benchmark_returns = benchmark_returns.reindex(returns.index).fillna(0)
        else:
            benchmark_returns = None
        
        # Create evaluator
        evaluator = StrategyEvaluator(returns, benchmark_returns)
        
        # Calculate performance metrics
        metrics = evaluator.get_summary_stats()
        
        # Create performance report
        try:
            figures = create_performance_report(returns, benchmark_returns)
            # Close figures to avoid memory leak
            for fig in figures:
                plt.close(fig)
        except Exception as e:
            print(f"Error creating performance report: {e}")
            figures = []
        
        # Compile results
        results = {
            'config': config.to_dict() if hasattr(config, 'to_dict') else vars(config),
            'metrics': metrics,
            'returns': returns,
            'signals': signals if isinstance(signals, pd.Series) else pd.Series(index=data.index),
            'portfolio': backtest_results,
            'trades': backtest_results.get('trades', pd.DataFrame()),
            'benchmark_returns': benchmark_returns,
        }
        
        return results
    
    except Exception as e:
        print(f"Error in backtest: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_results(results, output_dir, strategy_name):
    """
    Save backtest results
    
    Parameters:
    results (dict): Backtest results
    output_dir (str): Output directory
    strategy_name (str): Strategy name
    
    Returns:
    bool: True if successful, False otherwise
    """
    if results is None:
        print("No results to save")
        return False
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results
        filename = os.path.join(output_dir, f"{strategy_name}_{timestamp}")
        
        # Save to pickle (most complete format)
        pickle_path = f"{filename}.pickle"
        save_backtest_results(results, pickle_path, format='pickle')
        
        # Save metrics to JSON
        json_path = f"{filename}_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(results['metrics'], f, indent=4)
        
        # Save returns and portfolio to CSV
        for key in ['returns', 'signals']:
            if key in results and isinstance(results[key], pd.Series):
                csv_path = f"{filename}_{key}.csv"
                results[key].to_csv(csv_path)
        
        # Save portfolio DataFrame to CSV
        if 'portfolio' in results and isinstance(results['portfolio'], pd.DataFrame):
            csv_path = f"{filename}_portfolio.csv"
            results['portfolio'].to_csv(csv_path)
        
        # Save trades to CSV
        if 'trades' in results and isinstance(results['trades'], pd.DataFrame) and not results['trades'].empty:
            csv_path = f"{filename}_trades.csv"
            results['trades'].to_csv(csv_path)
        
        print(f"Results saved to {output_dir}")
        return True
    
    except Exception as e:
        print(f"Error saving results: {e}")
        return False


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run quantitative trading strategy backtest')
    
    parser.add_argument('--strategy', type=str, required=True,
                       help='Path to strategy file')
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to price data file')
    
    parser.add_argument('--market', type=str, required=True,
                       choices=['a_shares', 'hk_stocks', 'us_market'],
                       help='Market type')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (optional)')
    
    parser.add_argument('--benchmark', type=str, default=None,
                       help='Path to benchmark data file (optional)')
    
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results (default: results)')
    
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date for backtest (format: YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for backtest (format: YYYY-MM-DD)')
    
    parser.add_argument('--initial-capital', type=float, default=None,
                       help='Initial capital for backtest')
    
    args = parser.parse_args()
    
    # Load strategy
    print(f"Loading strategy from {args.strategy}...")
    strategy = load_strategy(args.strategy)
    if strategy is None:
        return
    
    # Load config
    print(f"Loading config for {args.market}...")
    config = load_config(args.market, args.config)
    if config is None:
        return
    
    # Override config with command line arguments
    if args.start_date is not None:
        config.start_date = args.start_date
    if args.end_date is not None:
        config.end_date = args.end_date
    if args.initial_capital is not None:
        config.initial_capital = args.initial_capital
    
    # Load price data
    print(f"Loading price data from {args.data}...")
    data = load_data(args.data, args.market)
    if data is None:
        return
    
    # Load benchmark data if provided
    benchmark_data = None
    if args.benchmark is not None:
        print(f"Loading benchmark data from {args.benchmark}...")
        benchmark_data = load_data(args.benchmark, args.market)
    
    # Run backtest
    print("Running backtest...")
    results = run_backtest(strategy, data, config, benchmark_data)
    
    # Save results
    if results is not None:
        strategy_name = strategy.__class__.__name__
        save_results(results, args.output, strategy_name)
    
    print("Backtest completed")


if __name__ == "__main__":
    main()
# config.py
# Configuration for US market backtest

"""
This file contains configuration parameters specific to US market
for backtesting quantitative trading strategies.
"""

# Market trading hours
MARKET_OPEN_TIME = "09:30:00"
MARKET_CLOSE_TIME = "16:00:00"
PRE_MARKET_OPEN_TIME = "04:00:00"
AFTER_MARKET_CLOSE_TIME = "20:00:00"

# Trading calendar
TRADING_DAYS_PER_YEAR = 252  # Standard US market trading days

# Transaction costs
COMMISSION_RATE = 0.0005  # 0.05% commission rate (typical discount broker)
MIN_COMMISSION = 0.0  # No minimum commission for most discount brokers
SEC_FEE_RATE = 0.0000229  # SEC fee for sell orders (rate as of 2023)
FINRA_TAF_RATE = 0.000119  # FINRA trading activity fee (per share, capped at $5.95)
FINRA_TAF_CAP = 5.95  # Maximum FINRA fee per trade

# Default benchmark
DEFAULT_BENCHMARK = "SPY"  # S&P 500 ETF as benchmark

# Risk management parameters
POSITION_SIZE_LIMIT = 0.05  # Maximum 5% of portfolio in a single stock
MAX_DRAWDOWN_LIMIT = 0.25  # Stop trading if drawdown exceeds 25%
VOLATILITY_LOOKBACK_PERIOD = 20  # Period for calculating volatility

# Backtest default parameters
DEFAULT_INITIAL_CAPITAL = 100000  # USD
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2021-12-31"

# Default margin requirements
INITIAL_MARGIN_REQUIREMENT = 0.50  # 50% initial margin requirement for long positions
MAINTENANCE_MARGIN_REQUIREMENT = 0.25  # 25% maintenance margin requirement
SHORT_MARGIN_REQUIREMENT = 1.50  # 150% of share value for short positions

# Default options parameters
OPTIONS_COMMISSION = 0.65  # Per contract commission for options
OPTIONS_ASSIGNMENT_FEE = 5.00  # Fee for options assignment
OPTIONS_EXERCISE_FEE = 5.00  # Fee for options exercise

# Short selling parameters
STOCK_LOAN_FEE_EASY_TO_BORROW = 0.0003  # 0.03% annual fee for easy-to-borrow stocks
STOCK_LOAN_FEE_HARD_TO_BORROW = 0.05  # 5% annual fee for hard-to-borrow stocks


class USMarketBacktestConfig:
    """
    Configuration class for US market backtest
    """
    
    def __init__(self, 
                 initial_capital=DEFAULT_INITIAL_CAPITAL,
                 start_date=DEFAULT_START_DATE,
                 end_date=DEFAULT_END_DATE,
                 benchmark=DEFAULT_BENCHMARK,
                 commission_rate=COMMISSION_RATE,
                 position_size_limit=POSITION_SIZE_LIMIT,
                 max_drawdown_limit=MAX_DRAWDOWN_LIMIT,
                 use_margin=False,
                 allow_short_selling=False,
                 include_extended_hours=False):
        """
        Initialize backtest configuration
        
        Parameters:
        initial_capital (float): Initial capital for backtest
        start_date (str): Start date for backtest
        end_date (str): End date for backtest
        benchmark (str): Benchmark symbol
        commission_rate (float): Commission rate
        position_size_limit (float): Maximum position size as percentage of portfolio
        max_drawdown_limit (float): Maximum drawdown before stopping
        use_margin (bool): Whether to allow margin trading
        allow_short_selling (bool): Whether to allow short selling
        include_extended_hours (bool): Whether to include pre-market and after-hours trading
        """
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark = benchmark
        self.commission_rate = commission_rate
        self.position_size_limit = position_size_limit
        self.max_drawdown_limit = max_drawdown_limit
        self.use_margin = use_margin
        self.allow_short_selling = allow_short_selling
        self.include_extended_hours = include_extended_hours
        
        # Derived parameters
        self.market_open_time = MARKET_OPEN_TIME
        self.market_close_time = MARKET_CLOSE_TIME
        self.pre_market_open_time = PRE_MARKET_OPEN_TIME if include_extended_hours else MARKET_OPEN_TIME
        self.after_market_close_time = AFTER_MARKET_CLOSE_TIME if include_extended_hours else MARKET_CLOSE_TIME
        self.trading_days_per_year = TRADING_DAYS_PER_YEAR
        self.min_commission = MIN_COMMISSION
        
        # Margin parameters
        self.initial_margin_requirement = INITIAL_MARGIN_REQUIREMENT
        self.maintenance_margin_requirement = MAINTENANCE_MARGIN_REQUIREMENT
        self.short_margin_requirement = SHORT_MARGIN_REQUIREMENT
        
        # Fee parameters
        self.sec_fee_rate = SEC_FEE_RATE
        self.finra_taf_rate = FINRA_TAF_RATE
        self.finra_taf_cap = FINRA_TAF_CAP
        
        # Short selling parameters
        self.stock_loan_fee_easy_to_borrow = STOCK_LOAN_FEE_EASY_TO_BORROW
        self.stock_loan_fee_hard_to_borrow = STOCK_LOAN_FEE_HARD_TO_BORROW
    
    def calculate_transaction_cost(self, price, volume, is_buy=True, is_hard_to_borrow=False):
        """
        Calculate transaction cost for a trade
        
        Parameters:
        price (float): Stock price
        volume (int): Number of shares
        is_buy (bool): True if buy order, False if sell order
        is_hard_to_borrow (bool): True if the stock is hard to borrow (for short selling)
        
        Returns:
        float: Transaction cost
        """
        # Calculate trade value
        trade_value = price * volume
        
        # Calculate commission
        commission = max(trade_value * self.commission_rate, self.min_commission)
        
        # SEC fee (sell orders only)
        sec_fee = trade_value * self.sec_fee_rate if not is_buy else 0
        
        # FINRA TAF (both buy and sell)
        finra_taf = min(volume * self.finra_taf_rate, self.finra_taf_cap)
        
        # Stock loan fee for short selling (sell orders only, annualized)
        stock_loan_fee = 0
        if not is_buy and self.allow_short_selling:
            daily_fee_rate = (self.stock_loan_fee_hard_to_borrow if is_hard_to_borrow 
                             else self.stock_loan_fee_easy_to_borrow) / 365
            stock_loan_fee = trade_value * daily_fee_rate
        
        # Total transaction cost
        total_cost = commission + sec_fee + finra_taf + stock_loan_fee
        
        return total_cost
    
    def calculate_margin_requirements(self, position_value, is_short=False):
        """
        Calculate margin requirements for a position
        
        Parameters:
        position_value (float): Market value of the position
        is_short (bool): True if short position, False if long position
        
        Returns:
        tuple: (initial_margin, maintenance_margin)
        """
        if not self.use_margin:
            # If margin trading is not enabled, require full position value
            return position_value, position_value
        
        if is_short and not self.allow_short_selling:
            # If short selling is not allowed, require full position value
            return position_value, position_value
        
        if is_short:
            # Short position margin requirements
            initial_margin = position_value * self.short_margin_requirement
            maintenance_margin = position_value * self.maintenance_margin_requirement
        else:
            # Long position margin requirements
            initial_margin = position_value * self.initial_margin_requirement
            maintenance_margin = position_value * self.maintenance_margin_requirement
        
        return initial_margin, maintenance_margin
    
    def calculate_max_leverage(self):
        """
        Calculate maximum leverage based on margin requirements
        
        Returns:
        float: Maximum leverage
        """
        if not self.use_margin:
            return 1.0  # No leverage if margin trading is disabled
        
        # Maximum leverage is 1 / initial margin requirement
        return 1.0 / self.initial_margin_requirement
    
    def calculate_annual_margin_interest(self, margin_loan_amount, margin_rate=0.05):
        """
        Calculate annual margin interest for a margin loan
        
        Parameters:
        margin_loan_amount (float): Amount of margin loan
        margin_rate (float): Annual margin interest rate
        
        Returns:
        float: Annual margin interest
        """
        if not self.use_margin or margin_loan_amount <= 0:
            return 0.0  # No interest if no margin loan
        
        return margin_loan_amount * margin_rate
    
    def to_dict(self):
        """
        Convert configuration to dictionary
        
        Returns:
        dict: Configuration as dictionary
        """
        return {
            'initial_capital': self.initial_capital,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'benchmark': self.benchmark,
            'commission_rate': self.commission_rate,
            'position_size_limit': self.position_size_limit,
            'max_drawdown_limit': self.max_drawdown_limit,
            'use_margin': self.use_margin,
            'allow_short_selling': self.allow_short_selling,
            'include_extended_hours': self.include_extended_hours,
            'market_open_time': self.market_open_time,
            'market_close_time': self.market_close_time,
            'pre_market_open_time': self.pre_market_open_time,
            'after_market_close_time': self.after_market_close_time,
            'trading_days_per_year': self.trading_days_per_year,
            'min_commission': self.min_commission,
            'initial_margin_requirement': self.initial_margin_requirement,
            'maintenance_margin_requirement': self.maintenance_margin_requirement,
            'short_margin_requirement': self.short_margin_requirement,
            'sec_fee_rate': self.sec_fee_rate,
            'finra_taf_rate': self.finra_taf_rate,
            'finra_taf_cap': self.finra_taf_cap,
            'stock_loan_fee_easy_to_borrow': self.stock_loan_fee_easy_to_borrow,
            'stock_loan_fee_hard_to_borrow': self.stock_loan_fee_hard_to_borrow
        }


# Example usage
if __name__ == "__main__":
    # Create default configuration
    default_config = USMarketBacktestConfig()
    
    # Print configuration
    for key, value in default_config.to_dict().items():
        print(f"{key}: {value}")
    
    # Calculate transaction cost example
    price = 150.0  # USD
    volume = 100  # shares
    
    buy_cost = default_config.calculate_transaction_cost(price, volume, is_buy=True)
    sell_cost = default_config.calculate_transaction_cost(price, volume, is_buy=False)
    
    print(f"\nTransaction cost for buying {volume} shares at {price} USD: {buy_cost:.2f} USD")
    print(f"Transaction cost for selling {volume} shares at {price} USD: {sell_cost:.2f} USD")
    
    # Calculate margin requirements example
    position_value = 10000  # USD
    
    long_margin = default_config.calculate_margin_requirements(position_value)
    short_margin = default_config.calculate_margin_requirements(position_value, is_short=True)
    
    print(f"\nMargin requirements for long position of {position_value} USD: {long_margin}")
    print(f"Margin requirements for short position of {position_value} USD: {short_margin}")
    
    # Calculate maximum leverage example
    max_leverage = default_config.calculate_max_leverage()
    
    print(f"\nMaximum leverage: {max_leverage:.2f}x")
    
    # Calculate margin interest example
    margin_loan = 5000  # USD
    margin_rate = 0.05  # 5% annual rate
    
    margin_interest = default_config.calculate_annual_margin_interest(margin_loan, margin_rate)
    
    print(f"\nAnnual margin interest for loan of {margin_loan} USD at {margin_rate*100}%: {margin_interest:.2f} USD")
# config.py
# Configuration for A-shares backtest

"""
This file contains configuration parameters specific to A-shares market
for backtesting quantitative trading strategies.
"""

# Market trading hours
MARKET_OPEN_TIME = "09:30:00"
MARKET_CLOSE_TIME = "15:00:00"
MORNING_CLOSE_TIME = "11:30:00"
AFTERNOON_OPEN_TIME = "13:00:00"

# Trading calendar
TRADING_DAYS_PER_YEAR = 245  # Approximately

# Price limits
PRICE_LIMIT_PCT = 0.10  # Regular stocks: ±10%
ST_PRICE_LIMIT_PCT = 0.05  # ST stocks: ±5%

# Trading rules
T_PLUS_1 = True  # T+1 rule: Can't sell stocks on the same day they're bought

# Lot size
LOT_SIZE = 100  # Stocks must be traded in multiples of 100 shares

# Transaction costs
COMMISSION_RATE = 0.0005  # 0.05% commission rate (broker fee)
MIN_COMMISSION = 5.0  # Minimum commission per trade (CNY)
STAMP_TAX_RATE = 0.001  # 0.1% stamp tax rate (only for sell orders)
TRANSFER_FEE_RATE = 0.00002  # 0.002% transfer fee (Shanghai market)
TRANSFER_FEE_MIN = 1.0  # Minimum transfer fee (CNY)

# Stock exchanges
EXCHANGES = {
    "SH": "Shanghai Stock Exchange",
    "SZ": "Shenzhen Stock Exchange"
}

# Index codes
INDICES = {
    "000001.SH": "Shanghai Composite Index",
    "399001.SZ": "Shenzhen Component Index",
    "000300.SH": "CSI 300 Index",
    "000016.SH": "SSE 50 Index",
    "000905.SH": "CSI 500 Index",
    "399006.SZ": "ChiNext Index"
}

# Default benchmark
DEFAULT_BENCHMARK = "000300.SH"  # CSI 300 Index

# Risk management parameters
POSITION_SIZE_LIMIT = 0.10  # Maximum 10% of portfolio in a single stock
MAX_DRAWDOWN_LIMIT = 0.20  # Stop trading if drawdown exceeds 20%
VOLATILITY_LOOKBACK_PERIOD = 20  # Period for calculating volatility

# Backtest default parameters
DEFAULT_INITIAL_CAPITAL = 1000000  # CNY
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2021-12-31"


class ASharesBacktestConfig:
    """
    Configuration class for A-shares backtest
    """
    
    def __init__(self, 
                 initial_capital=DEFAULT_INITIAL_CAPITAL,
                 start_date=DEFAULT_START_DATE,
                 end_date=DEFAULT_END_DATE,
                 benchmark=DEFAULT_BENCHMARK,
                 commission_rate=COMMISSION_RATE,
                 stamp_tax_rate=STAMP_TAX_RATE,
                 price_limit_pct=PRICE_LIMIT_PCT,
                 t_plus_1=T_PLUS_1,
                 position_size_limit=POSITION_SIZE_LIMIT,
                 max_drawdown_limit=MAX_DRAWDOWN_LIMIT):
        """
        Initialize backtest configuration
        
        Parameters:
        initial_capital (float): Initial capital for backtest
        start_date (str): Start date for backtest
        end_date (str): End date for backtest
        benchmark (str): Benchmark index code
        commission_rate (float): Commission rate
        stamp_tax_rate (float): Stamp tax rate
        price_limit_pct (float): Price limit percentage
        t_plus_1 (bool): T+1 trading rule
        position_size_limit (float): Maximum position size as percentage of portfolio
        max_drawdown_limit (float): Maximum drawdown before stopping
        """
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark = benchmark
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.price_limit_pct = price_limit_pct
        self.t_plus_1 = t_plus_1
        self.position_size_limit = position_size_limit
        self.max_drawdown_limit = max_drawdown_limit
        
        # Derived parameters
        self.market_open_time = MARKET_OPEN_TIME
        self.market_close_time = MARKET_CLOSE_TIME
        self.trading_days_per_year = TRADING_DAYS_PER_YEAR
        self.lot_size = LOT_SIZE
        self.min_commission = MIN_COMMISSION
    
    def calculate_transaction_cost(self, price, volume, is_buy=True):
        """
        Calculate transaction cost for a trade
        
        Parameters:
        price (float): Stock price
        volume (int): Number of shares
        is_buy (bool): True if buy order, False if sell order
        
        Returns:
        float: Transaction cost
        """
        # Calculate trade value
        trade_value = price * volume
        
        # Calculate commission
        commission = max(trade_value * self.commission_rate, self.min_commission)
        
        # Calculate stamp tax (sell orders only)
        stamp_tax = trade_value * self.stamp_tax_rate if not is_buy else 0
        
        # Calculate transfer fee (Shanghai market only, simplified)
        transfer_fee = max(trade_value * TRANSFER_FEE_RATE, TRANSFER_FEE_MIN)
        
        # Total transaction cost
        total_cost = commission + stamp_tax + transfer_fee
        
        return total_cost
    
    def get_price_limits(self, prev_close, is_st=False):
        """
        Calculate upper and lower price limits
        
        Parameters:
        prev_close (float): Previous closing price
        is_st (bool): True if the stock is an ST stock
        
        Returns:
        tuple: (upper_limit, lower_limit)
        """
        # Use ST stock limit if is_st is True
        limit_pct = ST_PRICE_LIMIT_PCT if is_st else self.price_limit_pct
        
        # Calculate upper and lower limits
        upper_limit = prev_close * (1 + limit_pct)
        lower_limit = prev_close * (1 - limit_pct)
        
        return upper_limit, lower_limit
    
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
            'stamp_tax_rate': self.stamp_tax_rate,
            'price_limit_pct': self.price_limit_pct,
            't_plus_1': self.t_plus_1,
            'position_size_limit': self.position_size_limit,
            'max_drawdown_limit': self.max_drawdown_limit,
            'market_open_time': self.market_open_time,
            'market_close_time': self.market_close_time,
            'trading_days_per_year': self.trading_days_per_year,
            'lot_size': self.lot_size,
            'min_commission': self.min_commission
        }


# Example usage
if __name__ == "__main__":
    # Create default configuration
    default_config = ASharesBacktestConfig()
    
    # Print configuration
    for key, value in default_config.to_dict().items():
        print(f"{key}: {value}")
    
    # Calculate transaction cost example
    price = 10.5  # CNY
    volume = 1000  # shares
    
    buy_cost = default_config.calculate_transaction_cost(price, volume, is_buy=True)
    sell_cost = default_config.calculate_transaction_cost(price, volume, is_buy=False)
    
    print(f"\nTransaction cost for buying {volume} shares at {price} CNY: {buy_cost:.2f} CNY")
    print(f"Transaction cost for selling {volume} shares at {price} CNY: {sell_cost:.2f} CNY")
    
    # Calculate price limits example
    prev_close = 10.0  # CNY
    
    regular_limits = default_config.get_price_limits(prev_close)
    st_limits = default_config.get_price_limits(prev_close, is_st=True)
    
    print(f"\nRegular stock price limits for previous close of {prev_close} CNY: {regular_limits}")
    print(f"ST stock price limits for previous close of {prev_close} CNY: {st_limits}")
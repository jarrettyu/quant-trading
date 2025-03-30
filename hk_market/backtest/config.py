# config.py
# Configuration for Hong Kong market backtest

"""
This file contains configuration parameters specific to Hong Kong market
for backtesting quantitative trading strategies.
"""

# Market trading hours
MARKET_OPEN_TIME = "09:30:00"
MARKET_CLOSE_TIME = "16:00:00"
MORNING_CLOSE_TIME = "12:00:00"
AFTERNOON_OPEN_TIME = "13:00:00"

# Trading calendar
TRADING_DAYS_PER_YEAR = 250  # Approximately

# Lot sizes (common Hong Kong stocks have different lot sizes)
DEFAULT_LOT_SIZE = 1000  # Default lot size for most stocks
LOT_SIZES = {
    # Format: "ticker": lot_size
    "0001.HK": 500,    # CK Hutchison
    "0005.HK": 400,    # HSBC Holdings
    "0700.HK": 100,    # Tencent
    "1299.HK": 200,    # AIA
    "9988.HK": 100,    # Alibaba
    # Add more as needed
}

# Transaction costs
COMMISSION_RATE = 0.0007  # 0.07% commission rate (broker fee)
MIN_COMMISSION = 8.0  # Minimum commission per trade (HKD)
STAMP_TAX_RATE = 0.001  # 0.1% stamp tax
TRADING_FEE_RATE = 0.00005  # 0.005% trading fee
CLEARING_FEE_RATE = 0.00002  # 0.002% clearing fee
SFC_LEVY_RATE = 0.000027  # 0.0027% SFC levy

# Stock Connect parameters for Northbound trading
STOCK_CONNECT_COMMISSION_RATE = 0.00108  # 0.108% commission for Northbound trading
STOCK_CONNECT_TAX_RATE = 0.001  # 0.1% tax for mainland China

# Default benchmark
DEFAULT_BENCHMARK = "^HSI"  # Hang Seng Index

# Risk management parameters
POSITION_SIZE_LIMIT = 0.10  # Maximum 10% of portfolio in a single stock
MAX_DRAWDOWN_LIMIT = 0.20  # Stop trading if drawdown exceeds 20%
VOLATILITY_LOOKBACK_PERIOD = 20  # Period for calculating volatility

# Backtest default parameters
DEFAULT_INITIAL_CAPITAL = 1000000  # HKD
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2021-12-31"

# Exchange rates
DEFAULT_HKD_CNY_RATE = 0.88  # HKD to CNY exchange rate
DEFAULT_HKD_USD_RATE = 0.128  # HKD to USD exchange rate


class HKBacktestConfig:
    """
    Configuration class for Hong Kong market backtest
    """
    
    def __init__(self, 
                 initial_capital=DEFAULT_INITIAL_CAPITAL,
                 start_date=DEFAULT_START_DATE,
                 end_date=DEFAULT_END_DATE,
                 benchmark=DEFAULT_BENCHMARK,
                 commission_rate=COMMISSION_RATE,
                 stamp_tax_rate=STAMP_TAX_RATE,
                 position_size_limit=POSITION_SIZE_LIMIT,
                 max_drawdown_limit=MAX_DRAWDOWN_LIMIT,
                 is_stock_connect=False):
        """
        Initialize backtest configuration
        
        Parameters:
        initial_capital (float): Initial capital for backtest
        start_date (str): Start date for backtest
        end_date (str): End date for backtest
        benchmark (str): Benchmark index code
        commission_rate (float): Commission rate
        stamp_tax_rate (float): Stamp tax rate
        position_size_limit (float): Maximum position size as percentage of portfolio
        max_drawdown_limit (float): Maximum drawdown before stopping
        is_stock_connect (bool): Whether to use Stock Connect rules for Northbound trading
        """
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark = benchmark
        self.position_size_limit = position_size_limit
        self.max_drawdown_limit = max_drawdown_limit
        self.is_stock_connect = is_stock_connect
        
        # Set appropriate commission and tax rates based on trading type
        if is_stock_connect:
            self.commission_rate = STOCK_CONNECT_COMMISSION_RATE
            self.stamp_tax_rate = STOCK_CONNECT_TAX_RATE
        else:
            self.commission_rate = commission_rate
            self.stamp_tax_rate = stamp_tax_rate
        
        # Derived parameters
        self.market_open_time = MARKET_OPEN_TIME
        self.market_close_time = MARKET_CLOSE_TIME
        self.trading_days_per_year = TRADING_DAYS_PER_YEAR
        self.default_lot_size = DEFAULT_LOT_SIZE
        self.min_commission = MIN_COMMISSION
        self.trading_fee_rate = TRADING_FEE_RATE
        self.clearing_fee_rate = CLEARING_FEE_RATE
        self.sfc_levy_rate = SFC_LEVY_RATE
        self.hkd_cny_rate = DEFAULT_HKD_CNY_RATE
        self.hkd_usd_rate = DEFAULT_HKD_USD_RATE
    
    def get_lot_size(self, ticker):
        """
        Get lot size for a specific stock
        
        Parameters:
        ticker (str): Stock ticker
        
        Returns:
        int: Lot size
        """
        return LOT_SIZES.get(ticker, self.default_lot_size)
    
    def calculate_transaction_cost(self, price, volume, ticker, is_buy=True):
        """
        Calculate transaction cost for a trade
        
        Parameters:
        price (float): Stock price
        volume (int): Number of shares
        ticker (str): Stock ticker
        is_buy (bool): True if buy order, False if sell order
        
        Returns:
        float: Transaction cost
        """
        # Calculate trade value
        trade_value = price * volume
        
        # Calculate commission
        commission = max(trade_value * self.commission_rate, self.min_commission)
        
        # Calculate stamp tax (both buy and sell in HK, only sell in mainland)
        if self.is_stock_connect:
            stamp_tax = trade_value * self.stamp_tax_rate if not is_buy else 0
        else:
            stamp_tax = trade_value * self.stamp_tax_rate
        
        # Calculate trading fee
        trading_fee = trade_value * self.trading_fee_rate
        
        # Calculate clearing fee
        clearing_fee = trade_value * self.clearing_fee_rate
        
        # Calculate SFC levy
        sfc_levy = trade_value * self.sfc_levy_rate
        
        # Total transaction cost
        total_cost = commission + stamp_tax + trading_fee + clearing_fee + sfc_levy
        
        return total_cost
    
    def get_shares_from_lot(self, volume, ticker):
        """
        Convert lot-based quantity to number of shares
        
        Parameters:
        volume (int): Number of lots
        ticker (str): Stock ticker
        
        Returns:
        int: Number of shares
        """
        lot_size = self.get_lot_size(ticker)
        return volume * lot_size
    
    def get_lots_from_shares(self, shares, ticker):
        """
        Convert number of shares to lot-based quantity
        
        Parameters:
        shares (int): Number of shares
        ticker (str): Stock ticker
        
        Returns:
        int: Number of lots
        """
        lot_size = self.get_lot_size(ticker)
        return shares // lot_size
    
    def convert_currency(self, amount, from_currency, to_currency):
        """
        Convert amount between currencies
        
        Parameters:
        amount (float): Amount to convert
        from_currency (str): Source currency ('HKD', 'CNY', 'USD')
        to_currency (str): Target currency ('HKD', 'CNY', 'USD')
        
        Returns:
        float: Converted amount
        """
        if from_currency == to_currency:
            return amount
        
        if from_currency == 'HKD':
            if to_currency == 'CNY':
                return amount * self.hkd_cny_rate
            elif to_currency == 'USD':
                return amount * self.hkd_usd_rate
        
        elif from_currency == 'CNY':
            if to_currency == 'HKD':
                return amount / self.hkd_cny_rate
            elif to_currency == 'USD':
                return amount / self.hkd_cny_rate * self.hkd_usd_rate
        
        elif from_currency == 'USD':
            if to_currency == 'HKD':
                return amount / self.hkd_usd_rate
            elif to_currency == 'CNY':
                return amount / self.hkd_usd_rate * self.hkd_cny_rate
        
        # If conversion not supported, return original amount
        return amount
    
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
            'position_size_limit': self.position_size_limit,
            'max_drawdown_limit': self.max_drawdown_limit,
            'is_stock_connect': self.is_stock_connect,
            'market_open_time': self.market_open_time,
            'market_close_time': self.market_close_time,
            'trading_days_per_year': self.trading_days_per_year,
            'default_lot_size': self.default_lot_size,
            'min_commission': self.min_commission,
            'trading_fee_rate': self.trading_fee_rate,
            'clearing_fee_rate': self.clearing_fee_rate,
            'sfc_levy_rate': self.sfc_levy_rate,
            'hkd_cny_rate': self.hkd_cny_rate,
            'hkd_usd_rate': self.hkd_usd_rate
        }


# Example usage
if __name__ == "__main__":
    # Create default configuration
    default_config = HKBacktestConfig()
    
    # Print configuration
    for key, value in default_config.to_dict().items():
        print(f"{key}: {value}")
    
    # Calculate transaction cost example
    ticker = "0700.HK"  # Tencent
    price = 400.0  # HKD
    volume = 1000  # shares
    
    buy_cost = default_config.calculate_transaction_cost(price, volume, ticker, is_buy=True)
    sell_cost = default_config.calculate_transaction_cost(price, volume, ticker, is_buy=False)
    
    print(f"\nTransaction cost for buying {volume} shares of {ticker} at {price} HKD: {buy_cost:.2f} HKD")
    print(f"Transaction cost for selling {volume} shares of {ticker} at {price} HKD: {sell_cost:.2f} HKD")
    
    # Lot size example
    ticker_tencent = "0700.HK"
    ticker_hsbc = "0005.HK"
    
    tencent_lot_size = default_config.get_lot_size(ticker_tencent)
    hsbc_lot_size = default_config.get_lot_size(ticker_hsbc)
    
    print(f"\nLot size for {ticker_tencent}: {tencent_lot_size}")
    print(f"Lot size for {ticker_hsbc}: {hsbc_lot_size}")
    
    # Convert lots to shares example
    lots_tencent = 5
    shares_tencent = default_config.get_shares_from_lot(lots_tencent, ticker_tencent)
    
    print(f"\n{lots_tencent} lots of {ticker_tencent} = {shares_tencent} shares")
    
    # Currency conversion example
    amount_hkd = 100000
    amount_cny = default_config.convert_currency(amount_hkd, 'HKD', 'CNY')
    amount_usd = default_config.convert_currency(amount_hkd, 'HKD', 'USD')
    
    print(f"\n{amount_hkd} HKD = {amount_cny:.2f} CNY")
    print(f"{amount_hkd} HKD = {amount_usd:.2f} USD")
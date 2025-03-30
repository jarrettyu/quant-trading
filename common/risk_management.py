# risk_management.py
# Common risk management functions for quantitative trading strategies

import numpy as np
import pandas as pd
from scipy import stats
import warnings


class PositionSizer:
    """
    Position sizing strategies for quantitative trading
    """
    
    @staticmethod
    def fixed_dollar_amount(capital, amount_per_trade, price):
        """
        Calculate position size based on a fixed dollar amount per trade
        
        Parameters:
        capital (float): Available capital
        amount_per_trade (float): Dollar amount per trade
        price (float): Price of the asset
        
        Returns:
        int: Number of shares/contracts to trade
        """
        # Validate inputs
        if amount_per_trade > capital:
            warnings.warn(f"Amount per trade ({amount_per_trade}) exceeds available capital ({capital}). Using maximum available capital.")
            amount_per_trade = capital
        
        # Calculate position size
        position_size = int(amount_per_trade / price)
        
        return position_size
    
    @staticmethod
    def fixed_percentage(capital, percentage, price):
        """
        Calculate position size based on a fixed percentage of capital
        
        Parameters:
        capital (float): Available capital
        percentage (float): Percentage of capital per trade (0-1)
        price (float): Price of the asset
        
        Returns:
        int: Number of shares/contracts to trade
        """
        # Validate inputs
        if not 0 <= percentage <= 1:
            raise ValueError("Percentage must be between 0 and 1")
        
        # Calculate position size
        amount = capital * percentage
        position_size = int(amount / price)
        
        return position_size
    
    @staticmethod
    def equal_weight(capital, num_positions, price):
        """
        Calculate position size based on equal weight across positions
        
        Parameters:
        capital (float): Available capital
        num_positions (int): Number of positions in the portfolio
        price (float): Price of the asset
        
        Returns:
        int: Number of shares/contracts to trade
        """
        # Validate inputs
        if num_positions <= 0:
            raise ValueError("Number of positions must be positive")
        
        # Calculate position size
        amount = capital / num_positions
        position_size = int(amount / price)
        
        return position_size
    
    @staticmethod
    def volatility_based(capital, risk_percentage, price, volatility, lookback_period=20):
        """
        Calculate position size based on asset volatility
        
        Parameters:
        capital (float): Available capital
        risk_percentage (float): Percentage of capital at risk per trade (0-1)
        price (float): Price of the asset
        volatility (float): Daily volatility or Series of asset returns
        lookback_period (int): Period for calculating volatility (if volatility is a Series)
        
        Returns:
        int: Number of shares/contracts to trade
        """
        # Validate inputs
        if not 0 <= risk_percentage <= 1:
            raise ValueError("Risk percentage must be between 0 and 1")
        
        # Calculate volatility if a Series of returns is provided
        if isinstance(volatility, (pd.Series, list, np.ndarray)):
            if isinstance(volatility, (list, np.ndarray)):
                volatility = pd.Series(volatility)
            daily_volatility = volatility.tail(lookback_period).std()
        else:
            daily_volatility = volatility
        
        # Annualize volatility
        annual_volatility = daily_volatility * np.sqrt(252)
        
        # Calculate dollar risk amount
        risk_amount = capital * risk_percentage
        
        # Calculate position size based on volatility (using 1 std dev as risk)
        # This means a 1 std dev move against the position would result in a loss equal to risk_amount
        dollar_per_share_risk = price * daily_volatility
        position_size = int(risk_amount / dollar_per_share_risk)
        
        return position_size
    
    @staticmethod
    def kelly_criterion(capital, win_rate, win_loss_ratio, price):
        """
        Calculate position size based on the Kelly Criterion
        
        Parameters:
        capital (float): Available capital
        win_rate (float): Probability of winning (0-1)
        win_loss_ratio (float): Ratio of average win to average loss
        price (float): Price of the asset
        
        Returns:
        int: Number of shares/contracts to trade
        """
        # Validate inputs
        if not 0 <= win_rate <= 1:
            raise ValueError("Win rate must be between 0 and 1")
        if win_loss_ratio <= 0:
            raise ValueError("Win/loss ratio must be positive")
        
        # Calculate Kelly percentage
        kelly_percentage = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Apply half-Kelly for more conservative sizing
        kelly_percentage = max(0, kelly_percentage * 0.5)
        
        # Calculate position size
        amount = capital * kelly_percentage
        position_size = int(amount / price)
        
        return position_size
    
    @staticmethod
    def optimal_f(capital, max_loss_percentage, price):
        """
        Calculate position size based on the Optimal F method
        
        Parameters:
        capital (float): Available capital
        max_loss_percentage (float): Maximum loss percentage per share (0-1)
        price (float): Price of the asset
        
        Returns:
        int: Number of shares/contracts to trade
        """
        # Validate inputs
        if not 0 <= max_loss_percentage <= 1:
            raise ValueError("Maximum loss percentage must be between 0 and 1")
        
        # Calculate optimal f (reciprocal of maximum loss percentage)
        optimal_f = 1 / (1 / max_loss_percentage)
        
        # Use a more conservative value (typical range: 0.1 to 0.3)
        optimal_f = min(0.2, optimal_f)
        
        # Calculate position size
        amount = capital * optimal_f
        position_size = int(amount / price)
        
        return position_size


class RiskManager:
    """
    Risk management for quantitative trading strategies
    """
    
    def __init__(self, initial_capital):
        """
        Initialize the risk manager
        
        Parameters:
        initial_capital (float): Initial capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_capital = initial_capital
        self.positions = {}  # Dictionary to store positions: symbol -> {quantity, entry_price}
        self.trades = []  # List to store historical trades
        self.equity_curve = [initial_capital]  # List to store equity curve
    
    def update_capital(self, new_capital):
        """
        Update current capital
        
        Parameters:
        new_capital (float): New capital value
        """
        self.current_capital = new_capital
        self.max_capital = max(self.max_capital, new_capital)
        self.equity_curve.append(new_capital)
    
    def add_position(self, symbol, quantity, entry_price):
        """
        Add a new position
        
        Parameters:
        symbol (str): Symbol of the asset
        quantity (int): Quantity of the asset
        entry_price (float): Entry price of the asset
        """
        self.positions[symbol] = {
            'quantity': quantity,
            'entry_price': entry_price
        }
    
    def update_position(self, symbol, quantity, price):
        """
        Update an existing position
        
        Parameters:
        symbol (str): Symbol of the asset
        quantity (int): New quantity of the asset
        price (float): Current price of the asset
        """
        if symbol in self.positions:
            old_position = self.positions[symbol]
            
            # Calculate position value change
            old_value = old_position['quantity'] * old_position['entry_price']
            new_value = quantity * price
            
            # Update capital
            self.update_capital(self.current_capital - old_value + new_value)
            
            # Update position
            if quantity == 0:
                # Position closed
                self.close_position(symbol, price)
            else:
                # Position updated
                self.positions[symbol] = {
                    'quantity': quantity,
                    'entry_price': price
                }
    
    def close_position(self, symbol, exit_price):
        """
        Close an existing position
        
        Parameters:
        symbol (str): Symbol of the asset
        exit_price (float): Exit price of the asset
        
        Returns:
        float: Profit/loss of the trade
        """
        if symbol in self.positions:
            position = self.positions[symbol]
            
            # Calculate profit/loss
            pnl = (exit_price - position['entry_price']) * position['quantity']
            
            # Add trade to history
            self.trades.append({
                'symbol': symbol,
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl
            })
            
            # Update capital
            self.update_capital(self.current_capital + pnl)
            
            # Remove position
            del self.positions[symbol]
            
            return pnl
        
        return 0.0
    
    def calculate_drawdown(self):
        """
        Calculate current drawdown
        
        Returns:
        float: Current drawdown (0-1)
        """
        # Calculate current drawdown
        drawdown = (self.max_capital - self.current_capital) / self.max_capital
        
        return drawdown
    
    def calculate_max_drawdown(self):
        """
        Calculate maximum historical drawdown
        
        Returns:
        float: Maximum drawdown (0-1)
        """
        # Calculate equity curve
        equity_curve = np.array(self.equity_curve)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_curve)
        
        # Calculate drawdown
        drawdown = (running_max - equity_curve) / running_max
        
        # Calculate maximum drawdown
        max_drawdown = np.max(drawdown)
        
        return max_drawdown
    
    def calculate_portfolio_var(self, confidence_level=0.95, lookback_period=252):
        """
        Calculate portfolio Value at Risk (VaR)
        
        Parameters:
        confidence_level (float): Confidence level for VaR (0-1)
        lookback_period (int): Period for calculating VaR
        
        Returns:
        float: Value at Risk (VaR)
        """
        if len(self.equity_curve) < lookback_period:
            warnings.warn("Not enough data for VaR calculation")
            return None
        
        # Calculate daily returns
        equity_curve = np.array(self.equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Use recent returns based on lookback period
        recent_returns = returns[-lookback_period:]
        
        # Calculate VaR
        var = np.percentile(recent_returns, 100 * (1 - confidence_level))
        
        # Convert to dollar value
        var_dollar = self.current_capital * abs(var)
        
        return var_dollar
    
    def calculate_position_risk(self, symbol, current_price):
        """
        Calculate risk exposure for a specific position
        
        Parameters:
        symbol (str): Symbol of the asset
        current_price (float): Current price of the asset
        
        Returns:
        float: Dollar risk of the position
        """
        if symbol in self.positions:
            position = self.positions[symbol]
            
            # Calculate position value
            position_value = position['quantity'] * current_price
            
            # Calculate risk as percentage of portfolio
            risk_percentage = position_value / self.current_capital
            
            return position_value, risk_percentage
        
        return 0.0, 0.0
    
    def calculate_portfolio_risk(self, prices_dict):
        """
        Calculate risk exposure for the entire portfolio
        
        Parameters:
        prices_dict (dict): Dictionary of current prices for all assets
        
        Returns:
        dict: Dictionary of portfolio risk metrics
        """
        # Calculate total portfolio value
        portfolio_value = 0.0
        position_values = {}
        
        for symbol, position in self.positions.items():
            if symbol in prices_dict:
                price = prices_dict[symbol]
                position_value = position['quantity'] * price
                position_values[symbol] = position_value
                portfolio_value += position_value
        
        # Calculate risk metrics
        risk_metrics = {
            'portfolio_value': portfolio_value,
            'cash': self.current_capital - portfolio_value,
            'total_capital': self.current_capital,
            'leverage': portfolio_value / self.current_capital if self.current_capital > 0 else 0,
            'position_weights': {symbol: value / self.current_capital for symbol, value in position_values.items()}
        }
        
        return risk_metrics
    
    def check_risk_limits(self, max_position_size=0.2, max_drawdown=0.2, max_leverage=1.5):
        """
        Check if any risk limits are breached
        
        Parameters:
        max_position_size (float): Maximum position size as percentage of capital
        max_drawdown (float): Maximum allowable drawdown
        max_leverage (float): Maximum allowable leverage
        
        Returns:
        dict: Dictionary of risk limit breaches
        """
        # Check position size limits
        position_breaches = {}
        for symbol, position in self.positions.items():
            position_value = position['quantity'] * position['entry_price']
            position_size = position_value / self.current_capital
            
            if position_size > max_position_size:
                position_breaches[symbol] = {
                    'limit': max_position_size,
                    'current': position_size
                }
        
        # Check drawdown limit
        drawdown = self.calculate_drawdown()
        drawdown_breach = drawdown > max_drawdown
        
        # Check leverage limit
        portfolio_value = sum(position['quantity'] * position['entry_price'] for position in self.positions.values())
        leverage = portfolio_value / self.current_capital
        leverage_breach = leverage > max_leverage
        
        # Compile all breaches
        breaches = {
            'position_breaches': position_breaches,
            'drawdown_breach': {
                'limit': max_drawdown,
                'current': drawdown
            } if drawdown_breach else None,
            'leverage_breach': {
                'limit': max_leverage,
                'current': leverage
            } if leverage_breach else None
        }
        
        return breaches
    
    def get_trade_statistics(self):
        """
        Calculate statistics from historical trades
        
        Returns:
        dict: Dictionary of trade statistics
        """
        if not self.trades:
            return {
                'num_trades': 0,
                'win_rate': None,
                'avg_win': None,
                'avg_loss': None,
                'win_loss_ratio': None,
                'profit_factor': None,
                'total_pnl': 0.0
            }
        
        # Calculate trade metrics
        num_trades = len(self.trades)
        win_trades = [trade for trade in self.trades if trade['pnl'] > 0]
        loss_trades = [trade for trade in self.trades if trade['pnl'] < 0]
        
        win_rate = len(win_trades) / num_trades if num_trades > 0 else 0
        
        avg_win = np.mean([trade['pnl'] for trade in win_trades]) if win_trades else 0
        avg_loss = abs(np.mean([trade['pnl'] for trade in loss_trades])) if loss_trades else 0
        
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf
        
        total_profit = sum(trade['pnl'] for trade in win_trades)
        total_loss = abs(sum(trade['pnl'] for trade in loss_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else np.inf
        total_pnl = total_profit - total_loss
        
        return {
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl
        }
    
    def suggest_position_size(self, symbol, price, volatility=None, method='fixed_percentage', **kwargs):
        """
        Suggest position size based on risk management rules
        
        Parameters:
        symbol (str): Symbol of the asset
        price (float): Price of the asset
        volatility (float, optional): Volatility of the asset
        method (str): Position sizing method
        **kwargs: Additional parameters for position sizing method
        
        Returns:
        int: Suggested position size
        """
        # Create position sizer
        sizer = PositionSizer()
        
        # Get available capital (excluding existing positions)
        available_capital = self.current_capital
        for position in self.positions.values():
            available_capital -= position['quantity'] * position['entry_price']
        
        # Get trade statistics
        stats = self.get_trade_statistics()
        
        # Apply position sizing method
        if method == 'fixed_dollar_amount':
            # Default to 5% of capital if not specified
            amount_per_trade = kwargs.get('amount_per_trade', self.current_capital * 0.05)
            position_size = sizer.fixed_dollar_amount(available_capital, amount_per_trade, price)
        
        elif method == 'fixed_percentage':
            # Default to 5% if not specified
            percentage = kwargs.get('percentage', 0.05)
            position_size = sizer.fixed_percentage(available_capital, percentage, price)
        
        elif method == 'equal_weight':
            # Default to 5 positions if not specified
            num_positions = kwargs.get('num_positions', 5)
            position_size = sizer.equal_weight(available_capital, num_positions, price)
        
        elif method == 'volatility_based':
            # Default to 1% risk if not specified
            risk_percentage = kwargs.get('risk_percentage', 0.01)
            lookback_period = kwargs.get('lookback_period', 20)
            position_size = sizer.volatility_based(available_capital, risk_percentage, price, volatility, lookback_period)
        
        elif method == 'kelly_criterion':
            # Use trade statistics or defaults
            win_rate = kwargs.get('win_rate', stats['win_rate'] or 0.5)
            win_loss_ratio = kwargs.get('win_loss_ratio', stats['win_loss_ratio'] or 1.0)
            position_size = sizer.kelly_criterion(available_capital, win_rate, win_loss_ratio, price)
        
        elif method == 'optimal_f':
            # Default to 10% max loss if not specified
            max_loss_percentage = kwargs.get('max_loss_percentage', 0.1)
            position_size = sizer.optimal_f(available_capital, max_loss_percentage, price)
        
        else:
            # Default to fixed percentage method
            percentage = kwargs.get('percentage', 0.05)
            position_size = sizer.fixed_percentage(available_capital, percentage, price)
        
        return position_size
    
    def adjust_position_for_risk_limits(self, symbol, quantity, price, max_position_size=0.2):
        """
        Adjust position size to stay within risk limits
        
        Parameters:
        symbol (str): Symbol of the asset
        quantity (int): Proposed quantity
        price (float): Price of the asset
        max_position_size (float): Maximum position size as percentage of capital
        
        Returns:
        int: Adjusted position size
        """
        # Calculate position value
        position_value = quantity * price
        
        # Calculate position size as percentage of capital
        position_size = position_value / self.current_capital
        
        # Check if position size exceeds limit
        if position_size > max_position_size:
            # Adjust quantity to meet limit
            adjusted_quantity = int(self.current_capital * max_position_size / price)
            return adjusted_quantity
        
        return quantity


# Example usage
if __name__ == "__main__":
    # Create risk manager
    risk_manager = RiskManager(initial_capital=100000)
    
    # Add positions
    risk_manager.add_position('AAPL', 100, 150.0)
    risk_manager.add_position('MSFT', 50, 200.0)
    
    # Update capital
    risk_manager.update_capital(120000)
    
    # Close a position
    pnl = risk_manager.close_position('AAPL', 160.0)
    print(f"AAPL position closed with P&L: ${pnl:.2f}")
    
    # Calculate drawdown
    drawdown = risk_manager.calculate_drawdown()
    print(f"Current drawdown: {drawdown:.2%}")
    
    # Check risk limits
    breaches = risk_manager.check_risk_limits()
    print("Risk limit breaches:", breaches)
    
    # Suggest position size
    price = 300.0
    position_size = risk_manager.suggest_position_size('GOOG', price, method='fixed_percentage', percentage=0.1)
    print(f"Suggested position size for GOOG at ${price:.2f}: {position_size} shares")
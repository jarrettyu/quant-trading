# Quant Trading Strategies
A collaborative project exploring quantitave trading strategies across US, Hong Kong, and Chinese A-shares markets.

## Overview

This repository contains implementations of various quantitative trading strategies across different markets:

- **A-Shares Market**: Strategies for China's mainland stock market
- **Hong Kong Market**: Strategies for Hong Kong-listed stocks
- **US Market**: Strategies for US equities

The goal is to learn quantitative trading principles by implementing basic strategies and testing them on historical data.

## Implemented Strategies

### A-Shares Market
- **Double Moving Average**: Classic MA crossover strategy adapted for A-shares with T+1 trading rules
- **RSI Strategy**: Relative Strength Index based mean-reversion strategy

### Hong Kong Market
- **AH Premium**: Arbitrage strategy exploiting price differences between A-shares and H-shares of dual-listed companies
- **Northbound Flow**: Strategy tracking capital flows from mainland China through Stock Connect

### US Market
- **Bollinger Bands**: Mean reversion strategy using Bollinger Bands
- **Momentum Factor**: Factor-based momentum strategy with periodic rebalancing

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai4fin-quantitative-trading.git
cd ai4fin-quantitative-trading
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running a Backtest
```bash
python backtest.py --strategy a_shares/strategies/double_moving_average.py --data path/to/data.csv --market a_shares
```

## Contributing

This is a learning repository - contributions, improvements, and discussions are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-strategy`)
3. Commit your changes (`git commit -m 'Add some amazing strategy'`)
4. Push to the branch (`git push origin feature/amazing-strategy`)
5. Open a Pull Request

## Disclaimer

This repository is for educational purposes only. The strategies are meant for learning and not as investment advice. Always do your own research before actual trading.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
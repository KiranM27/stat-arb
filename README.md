# Cryptocurrency Statistical Arbitrage

A statistical arbitrage trading strategy implementation using cointegrated cryptocurrency pairs. This project identifies cointegrated pairs of assets and generates trading signals based on mean reversion principles.

## Overview

Statistical arbitrage is a trading strategy that exploits temporary price divergences between historically correlated assets. This implementation focuses on cryptocurrency pairs trading:

1. **Cointegration Analysis**: Identifies cryptocurrency pairs that exhibit a long-term statistical relationship
2. **Signal Generation**: Creates trading signals based on deviations from the long-term equilibrium (Z-scores)
3. **Backtesting**: Evaluates strategy performance over historical data

## Project Structure

```
├── data_analysis/
│   ├── signal_generation.py    # Signal generation and backtesting logic
│   └── ...                     # Other analysis modules
├── cointegration_results/      # Saved cointegration test results
├── cleaned_ohlcv_data/         # Price data for cryptocurrencies
├── backtest_results/           # Backtest output (plots, performance metrics)
├── pair_plots/                 # Visualizations of cointegrated pairs
├── run_signal_generator.py     # Runner script for signal generation
└── ...
```

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd stat-arb
   ```

2. Set up a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install pandas numpy matplotlib statsmodels
   ```

## Usage

### Running the Signal Generator

To run the signal generator with default parameters:

```bash
python run_signal_generator.py
```

With custom parameters:

```bash
python run_signal_generator.py --entry 1.5 --exit 0.75 --stop-loss 3.0 --max-holding 3.0 --position-size 10000 --top-n 3
```

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--entry` | Z-score threshold for position entry | 2.0 |
| `--exit` | Z-score threshold for position exit | 0.5 |
| `--stop-loss` | Z-score threshold for stop loss | 3.0 |
| `--max-holding` | Maximum holding period as a factor of half-life | 3.0 |
| `--position-size` | Notional position size in dollars | 10000 |
| `--max-drawdown` | Maximum allowable drawdown before stopping trading | 0.15 (15%) |
| `--top-n` | Number of top cointegrated pairs to backtest | 5 |
| `--lookback-days` | Number of days to use for backtesting | None (all available data) |

## Trading Strategy

This implementation follows a mean-reversion strategy based on the Z-score of the spread between two cointegrated assets:

### Trading Rules

1. **Entry Conditions**:
   - When Z-score ≥ +entry_threshold: SHORT spread (short Y, long X)
   - When Z-score ≤ -entry_threshold: LONG spread (long Y, short X)

2. **Exit Conditions**:
   - When |Z-score| ≤ exit_threshold (profit target)
   - When |Z-score| ≥ stop_loss_threshold (stop loss)
   - After max_holding_periods * half_life days (time-based exit)

3. **Risk Management**:
   - Fixed notional position sizing
   - Maximum drawdown limit
   - Trade-level stop losses

### Key Concepts

- **Cointegration**: A statistical property where two time series maintain a long-term equilibrium relationship
- **Spread**: The difference between two assets after accounting for their statistical relationship
- **Z-score**: A normalized measure of how many standard deviations the spread is from its mean
- **Half-life**: The estimated time for the spread to revert halfway back to its mean

## Interpretation of Results

Backtest results are saved to the `backtest_results/` directory:

### CSV Summary Files

CSV files contain performance metrics for each pair:
- Total P&L
- Number of trades
- Win rate
- Average win/loss
- Profit factor
- Maximum drawdown

### Visualization Plots

For each pair, a plot is generated with:
1. Normalized asset prices
2. Z-score with entry/exit thresholds
3. Position indicator (long/short/flat)
4. Cumulative P&L

## Examples

### Basic Backtest

```bash
python run_signal_generator.py --top-n 3
```

This runs a backtest on the top 3 cointegrated pairs with default parameters.

### Recent Data Backtest

```bash
python run_signal_generator.py --lookback-days 60
```

This runs a backtest using only the last 60 days of data.

### Customized Strategy

```bash
python run_signal_generator.py --entry 1.8 --exit 0.6 --stop-loss 2.5 --max-holding 2.0 --position-size 5000
```

This adjusts the strategy to be more conservative with earlier entries/exits and smaller position sizes.

## Performance Optimization

To optimize the strategy, consider:

1. Adjusting Z-score thresholds based on volatility
2. Position sizing based on pair stability
3. Implementing volatility-based filters
4. Using asymmetric entry/exit rules

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
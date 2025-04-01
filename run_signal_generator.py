#!/usr/bin/env python
import sys
import os

# Add data_analysis to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_analysis_dir = os.path.join(script_dir, 'data_analysis')
sys.path.append(data_analysis_dir)

# Import signal generator
from data_analysis.signal_generation import StatArbSignalGenerator

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate trading signals and backtest statistical arbitrage strategies')
    parser.add_argument('--entry', type=float, default=2.0, 
                        help='Z-score threshold for entry (default: 2.0)')
    parser.add_argument('--exit', type=float, default=0.5,
                        help='Z-score threshold for exit (default: 0.5)')
    parser.add_argument('--stop-loss', type=float, default=3.0,
                        help='Z-score threshold for stop loss (default: 3.0)')
    parser.add_argument('--max-holding', type=float, default=3.0,
                        help='Maximum holding period as a factor of half-life (default: 3.0)')
    parser.add_argument('--position-size', type=float, default=10000,
                        help='Notional position size in dollars (default: $10,000)')
    parser.add_argument('--max-drawdown', type=float, default=0.15,
                        help='Maximum allowable drawdown before stopping trading (default: 15%)')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of top pairs to backtest (default: 5)')
    parser.add_argument('--lookback-days', type=int, default=None,
                        help='Number of days to use for backtesting (default: None, use all available data)')
    
    args = parser.parse_args()
    
    print("Starting signal generation and backtesting...")
    
    # Create signal generator
    signal_gen = StatArbSignalGenerator(
        entry_threshold=args.entry,
        exit_threshold=args.exit,
        stop_loss_threshold=args.stop_loss,
        max_holding_periods=args.max_holding,
        position_size=args.position_size,
        max_drawdown_pct=args.max_drawdown
    )
    
    # Run backtest
    result = signal_gen.run_backtest_on_all_pairs(
        top_n=args.top_n,
        lookback_days=args.lookback_days
    )
    
    if result:
        print("Backtesting completed successfully.")
    else:
        print("Error occurred during backtesting.") 
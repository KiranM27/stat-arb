#!/usr/bin/env python
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import statsmodels.api as sm

class StatArbSignalGenerator:
    """
    Signal generator for statistical arbitrage based on cointegrated pairs
    
    Trading Rules:
    1. Entry when |Z-score| >= entry_threshold (default: 2.0)
       - If Z >= +entry_threshold: short spread (short Y, long X)
       - If Z <= -entry_threshold: long spread (long Y, short X)
    2. Exit when |Z-score| <= exit_threshold (default: 0.5)
    3. Stop loss when |Z-score| >= stop_loss_threshold (default: 3.0)
    4. Time-based exit after max_holding_periods * half_life
    """
    
    def __init__(self, 
                 entry_threshold=2.0, 
                 exit_threshold=0.5, 
                 stop_loss_threshold=3.0,
                 max_holding_periods=3,
                 position_size=10000,
                 max_drawdown_pct=0.15):
        """
        Initialize the signal generator with trading parameters
        
        Parameters:
        entry_threshold: Z-score threshold for entry (default: 2.0)
        exit_threshold: Z-score threshold for exit (default: 0.5)
        stop_loss_threshold: Z-score threshold for stop loss (default: 3.0)
        max_holding_periods: Maximum holding period as a factor of half-life (default: 3)
        position_size: Notional position size in dollars (default: $10,000)
        max_drawdown_pct: Maximum allowable drawdown before stopping trading (default: 15%)
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.max_holding_periods = max_holding_periods
        self.position_size = position_size
        self.max_drawdown_pct = max_drawdown_pct
        
        # Store loaded data
        self.price_data = {}
        self.cointegrated_pairs = None
        
    def load_price_data(self, data_dir='cleaned_ohlcv_data'):
        """Load price data for all available tokens"""
        if not os.path.exists(data_dir):
            print(f"Error: Data directory {data_dir} does not exist!")
            return False
        
        # Dictionary to store price dataframes
        self.price_data = {}
        
        for file_name in os.listdir(data_dir):
            if not file_name.endswith('_USDT_daily.csv'):
                continue
            
            symbol = file_name.split('_')[0]
            file_path = os.path.join(data_dir, file_name)
            
            try:
                # Read the data
                df = pd.read_csv(file_path)
                
                # Convert timestamp to datetime if it's string
                if isinstance(df['timestamp'].iloc[0], str):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Sort by timestamp
                df = df.sort_values('timestamp')
                
                # Set timestamp as index and keep OHLCV data
                self.price_data[symbol] = df.set_index('timestamp')
                
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
        
        print(f"Loaded price data for {len(self.price_data)} tokens")
        return True
        
    def load_cointegrated_pairs(self, file_path=None):
        """
        Load cointegrated pairs from file
        
        If file_path is None, use the most recent file in cointegration_results directory
        """
        if file_path is None:
            # Find the most recent file
            result_dir = 'cointegration_results'
            if not os.path.exists(result_dir):
                print(f"Error: Results directory {result_dir} does not exist!")
                return False
            
            # Find cointegrated pairs files
            pair_files = [f for f in os.listdir(result_dir) 
                         if f.startswith('cointegrated_pairs_')]
            
            if not pair_files:
                print("No cointegrated pairs files found")
                return False
            
            # Get the most recent file
            latest_file = sorted(pair_files)[-1]
            file_path = os.path.join(result_dir, latest_file)
        
        # Load the cointegrated pairs
        try:
            self.cointegrated_pairs = pd.read_csv(file_path)
            print(f"Loaded {len(self.cointegrated_pairs)} cointegrated pairs from {file_path}")
            return True
        except Exception as e:
            print(f"Error loading cointegrated pairs: {e}")
            return False
            
    def generate_signals(self, pair_index=0, lookback_days=None):
        """
        Generate trading signals for a specified pair
        
        Parameters:
        pair_index: Index of the pair in the cointegrated_pairs DataFrame (default: 0)
        lookback_days: Number of days to use for backtesting (default: None, use all available data)
        
        Returns:
        DataFrame with signals and performance metrics
        """
        if self.cointegrated_pairs is None or len(self.cointegrated_pairs) <= pair_index:
            print("Error: Cointegrated pairs not loaded or invalid pair index")
            return None
        
        # Get pair information
        pair = self.cointegrated_pairs.iloc[pair_index]
        symbol1 = pair['Symbol1']  # X in the equation (hedge_ratio * X + intercept = Y)
        symbol2 = pair['Symbol2']  # Y in the equation
        hedge_ratio = pair['Hedge_Ratio']
        intercept = pair['Intercept']
        half_life = pair['Half_Life']
        
        print(f"Generating signals for pair: {symbol1}-{symbol2}")
        print(f"  Hedge ratio: {hedge_ratio:.6f}")
        print(f"  Intercept: {intercept:.6f}")
        print(f"  Half-life: {half_life:.2f} days")
        
        # Ensure both symbols are in price data
        if symbol1 not in self.price_data or symbol2 not in self.price_data:
            print(f"Error: Price data not available for {symbol1} or {symbol2}")
            return None
        
        # Get price series
        price1 = self.price_data[symbol1]['close']
        price2 = self.price_data[symbol2]['close']
        
        # Align dates
        common_dates = price1.index.intersection(price2.index)
        price1 = price1.loc[common_dates]
        price2 = price2.loc[common_dates]
        
        # Sort by date
        price1 = price1.sort_index()
        price2 = price2.sort_index()
        
        # Limit to lookback period if specified
        if lookback_days is not None:
            cutoff_date = price1.index[-1] - pd.Timedelta(days=lookback_days)
            price1 = price1[price1.index >= cutoff_date]
            price2 = price2[price2.index >= cutoff_date]
        
        # Calculate spread
        spread = price2 - (intercept + hedge_ratio * price1)
        
        # Calculate rolling z-score
        window = min(int(half_life * 3), 30)  # Use 3x half-life or 30 days, whichever is smaller
        spread_mean = spread.rolling(window=window, min_periods=5).mean()
        spread_std = spread.rolling(window=window, min_periods=5).std()
        z_score = (spread - spread_mean) / spread_std
        
        # Initialize signal dataframe
        signals = pd.DataFrame(index=spread.index)
        signals['price1'] = price1
        signals['price2'] = price2
        signals['spread'] = spread
        signals['z_score'] = z_score
        signals['position'] = 0  # 1: long spread, -1: short spread, 0: no position
        signals['entry_price1'] = np.nan
        signals['entry_price2'] = np.nan
        signals['exit_price1'] = np.nan
        signals['exit_price2'] = np.nan
        signals['trade_pnl'] = 0.0
        signals['cumulative_pnl'] = 0.0
        signals['drawdown'] = 0.0
        signals['entry_date'] = pd.NaT
        signals['days_in_trade'] = 0
        
        # Generate signals
        in_position = 0
        entry_date = None
        max_cumulative_pnl = 0
        stopped_trading = False
        
        for i in range(window, len(signals)):
            today = signals.index[i]
            z = signals.loc[today, 'z_score']
            
            if np.isnan(z):
                continue
                
            # Skip if maximum drawdown exceeded
            if signals.loc[signals.index[i-1], 'drawdown'] <= -self.max_drawdown_pct and not stopped_trading:
                print(f"Max drawdown exceeded on {today.strftime('%Y-%m-%d')}. Stopping trading.")
                stopped_trading = True
            
            # Check for entry signals
            if in_position == 0 and not stopped_trading:
                # Long spread signal (z-score too negative)
                if z <= -self.entry_threshold:
                    in_position = 1
                    entry_date = today
                    signals.loc[today, 'position'] = 1
                    signals.loc[today, 'entry_price1'] = signals.loc[today, 'price1']
                    signals.loc[today, 'entry_price2'] = signals.loc[today, 'price2']
                    signals.loc[today, 'entry_date'] = today
                # Short spread signal (z-score too positive)
                elif z >= self.entry_threshold:
                    in_position = -1
                    entry_date = today
                    signals.loc[today, 'position'] = -1
                    signals.loc[today, 'entry_price1'] = signals.loc[today, 'price1']
                    signals.loc[today, 'entry_price2'] = signals.loc[today, 'price2']
                    signals.loc[today, 'entry_date'] = today
            
            # Check for exit signals for existing position
            elif in_position != 0:
                # Track days in trade
                if entry_date is not None:
                    signals.loc[today, 'days_in_trade'] = (today - entry_date).days
                
                exit_signal = False
                exit_reason = ""
                
                # Exit when z-score reverts
                if (in_position == 1 and z >= -self.exit_threshold) or \
                   (in_position == -1 and z <= self.exit_threshold):
                    exit_signal = True
                    exit_reason = "Target reached"
                
                # Stop loss when z-score moves against position
                elif (in_position == 1 and z < -self.stop_loss_threshold) or \
                     (in_position == -1 and z > self.stop_loss_threshold):
                    exit_signal = True
                    exit_reason = "Stop loss"
                
                # Time-based exit
                max_holding_days = int(half_life * self.max_holding_periods)
                if entry_date is not None and (today - entry_date).days >= max_holding_days:
                    exit_signal = True
                    exit_reason = "Time-based exit"
                
                # Execute exit if signal triggered
                if exit_signal:
                    signals.loc[today, 'position'] = 0
                    signals.loc[today, 'exit_price1'] = signals.loc[today, 'price1']
                    signals.loc[today, 'exit_price2'] = signals.loc[today, 'price2']
                    
                    # Calculate trade P&L
                    entry_price1 = signals.loc[signals['entry_date'] == entry_date, 'entry_price1'].iloc[0]
                    entry_price2 = signals.loc[signals['entry_date'] == entry_date, 'entry_price2'].iloc[0]
                    exit_price1 = signals.loc[today, 'exit_price1']
                    exit_price2 = signals.loc[today, 'exit_price2']
                    
                    # Position sizing based on fixed notional value
                    value1 = self.position_size
                    value2 = self.position_size
                    
                    # Calculate quantity based on entry prices
                    qty1 = value1 / entry_price1
                    qty2 = value2 / entry_price2
                    
                    if in_position == 1:  # Long spread (long y, short x)
                        # Long y (symbol2)
                        pnl_y = qty2 * (exit_price2 - entry_price2)
                        # Short x (symbol1)
                        pnl_x = qty1 * (entry_price1 - exit_price1)
                    else:  # Short spread (short y, long x)
                        # Short y (symbol2)
                        pnl_y = qty2 * (entry_price2 - exit_price2)
                        # Long x (symbol1)
                        pnl_x = qty1 * (exit_price1 - entry_price1)
                    
                    pnl = pnl_x + pnl_y
                    signals.loc[today, 'trade_pnl'] = pnl
                    print(f"Exit on {today.strftime('%Y-%m-%d')}: {exit_reason}. PnL: ${pnl:.2f}")
                    
                    # Reset trade tracking
                    in_position = 0
                    entry_date = None
                else:
                    # Maintain position
                    signals.loc[today, 'position'] = in_position
                    signals.loc[today, 'entry_date'] = entry_date
            
            # Calculate cumulative PnL
            if i > window:
                signals.loc[today, 'cumulative_pnl'] = signals.loc[signals.index[i-1], 'cumulative_pnl'] + signals.loc[today, 'trade_pnl']
            else:
                signals.loc[today, 'cumulative_pnl'] = signals.loc[today, 'trade_pnl']
            
            # Update maximum cumulative PnL for drawdown calculation
            if signals.loc[today, 'cumulative_pnl'] > max_cumulative_pnl:
                max_cumulative_pnl = signals.loc[today, 'cumulative_pnl']
            
            # Calculate drawdown
            if max_cumulative_pnl > 0:
                drawdown = (signals.loc[today, 'cumulative_pnl'] - max_cumulative_pnl) / max_cumulative_pnl
            else:
                drawdown = 0
            signals.loc[today, 'drawdown'] = drawdown
        
        # Calculate performance metrics
        total_pnl = signals['trade_pnl'].sum()
        winning_trades = signals[signals['trade_pnl'] > 0]['trade_pnl'].count()
        losing_trades = signals[signals['trade_pnl'] < 0]['trade_pnl'].count()
        total_trades = winning_trades + losing_trades
        
        if total_trades > 0:
            win_rate = winning_trades / total_trades
            avg_win = signals[signals['trade_pnl'] > 0]['trade_pnl'].mean() if winning_trades > 0 else 0
            avg_loss = signals[signals['trade_pnl'] < 0]['trade_pnl'].mean() if losing_trades > 0 else 0
            profit_factor = abs(signals[signals['trade_pnl'] > 0]['trade_pnl'].sum() / signals[signals['trade_pnl'] < 0]['trade_pnl'].sum()) if signals[signals['trade_pnl'] < 0]['trade_pnl'].sum() != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        max_drawdown = signals['drawdown'].min()
        
        # Print performance summary
        print(f"\nPerformance Summary for {symbol1}-{symbol2}:")
        print(f"Total PnL: ${total_pnl:.2f}")
        print(f"Total trades: {total_trades}")
        print(f"Win rate: {win_rate:.2%}")
        print(f"Average win: ${avg_win:.2f}")
        print(f"Average loss: ${avg_loss:.2f}")
        print(f"Profit factor: {profit_factor:.2f}")
        print(f"Maximum drawdown: {max_drawdown:.2%}")
        
        # Store metrics in a dictionary
        metrics = {
            "pair": f"{symbol1}-{symbol2}",
            "hedge_ratio": hedge_ratio,
            "intercept": intercept,
            "half_life": half_life,
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown
        }
        
        return signals, metrics
    
    def plot_backtest_results(self, signals, pair_info, save_dir='backtest_results'):
        """
        Plot the backtest results
        
        Parameters:
        signals: DataFrame with signal and performance data
        pair_info: Dictionary with pair information
        save_dir: Directory to save plots
        """
        if signals is None or len(signals) == 0:
            print("No signals to plot")
            return
        
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        symbol1 = pair_info['pair'].split('-')[0]
        symbol2 = pair_info['pair'].split('-')[1]
        hedge_ratio = pair_info['hedge_ratio']
        
        # Plot prices, spread, z-score, signals, and PnL
        fig, axes = plt.subplots(4, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [2, 1, 1, 1]})
        
        # Plot 1: Normalized prices
        norm_price1 = signals['price1'] / signals['price1'].iloc[0]
        norm_price2 = signals['price2'] / signals['price2'].iloc[0]
        
        axes[0].plot(signals.index, norm_price1, label=f'{symbol1} (normalized)')
        axes[0].plot(signals.index, norm_price2, label=f'{symbol2} (normalized)')
        axes[0].set_title(f'Normalized Prices: {symbol1} vs {symbol2}')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot 2: Z-score and entry/exit zones
        axes[1].plot(signals.index, signals['z_score'])
        axes[1].axhline(y=0, color='black', linestyle='-')
        axes[1].axhline(y=self.entry_threshold, color='r', linestyle='--', label=f'Entry (+{self.entry_threshold})')
        axes[1].axhline(y=-self.entry_threshold, color='g', linestyle='--', label=f'Entry (-{self.entry_threshold})')
        axes[1].axhline(y=self.exit_threshold, color='b', linestyle=':', label=f'Exit (+{self.exit_threshold})')
        axes[1].axhline(y=-self.exit_threshold, color='b', linestyle=':', label=f'Exit (-{self.exit_threshold})')
        axes[1].axhline(y=self.stop_loss_threshold, color='m', linestyle='-.', label=f'Stop Loss (+{self.stop_loss_threshold})')
        axes[1].axhline(y=-self.stop_loss_threshold, color='m', linestyle='-.', label=f'Stop Loss (-{self.stop_loss_threshold})')
        axes[1].set_title(f'Z-Score (Hedge Ratio: {hedge_ratio:.4f})')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot 3: Positions
        axes[2].plot(signals.index, signals['position'])
        axes[2].set_yticks([-1, 0, 1])
        axes[2].set_yticklabels(['Short Spread', 'No Position', 'Long Spread'])
        axes[2].set_title('Position')
        axes[2].grid(True)
        
        # Plot 4: Cumulative PnL
        axes[3].plot(signals.index, signals['cumulative_pnl'])
        axes[3].set_title('Cumulative P&L ($)')
        axes[3].grid(True)
        
        # Add performance metrics in a text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = '\n'.join((
            f"Total PnL: ${pair_info['total_pnl']:.2f}",
            f"Trades: {pair_info['total_trades']}",
            f"Win Rate: {pair_info['win_rate']:.2%}",
            f"Profit Factor: {pair_info['profit_factor']:.2f}",
            f"Max Drawdown: {pair_info['max_drawdown']:.2%}"
        ))
        axes[3].text(0.05, 0.95, textstr, transform=axes[3].transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f"{save_dir}/{symbol1}_{symbol2}_backtest_{timestamp}.png", dpi=300)
        plt.close()
        
        print(f"Saved backtest plot to {save_dir}/{symbol1}_{symbol2}_backtest_{timestamp}.png")
    
    def run_backtest_on_all_pairs(self, top_n=5, lookback_days=None):
        """
        Run backtest on all pairs or top N pairs
        
        Parameters:
        top_n: Number of top pairs to backtest (default: 5)
        lookback_days: Number of days to use for backtesting (default: None, use all available data)
        
        Returns:
        Dictionary with backtest results
        """
        if not self.load_price_data():
            return None
        
        if not self.load_cointegrated_pairs():
            return None
        
        # Limit to top N pairs
        n_pairs = min(top_n, len(self.cointegrated_pairs))
        
        # Store results
        all_results = {}
        
        for i in range(n_pairs):
            print(f"\nBacktesting pair {i+1}/{n_pairs}...")
            signals, metrics = self.generate_signals(pair_index=i, lookback_days=lookback_days)
            
            if signals is not None:
                # Plot results
                self.plot_backtest_results(signals, metrics)
                
                # Store results
                pair_name = metrics['pair']
                all_results[pair_name] = metrics
        
        # Save summary to file
        if all_results:
            results_df = pd.DataFrame.from_dict(all_results, orient='index')
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_df.to_csv(f"backtest_results/backtest_summary_{timestamp}.csv")
            
            print("\nBacktest Summary:")
            print(results_df[['total_pnl', 'total_trades', 'win_rate', 'profit_factor', 'max_drawdown']])
        
        return all_results

def main():
    """Run signal generation and backtesting"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate trading signals for statistical arbitrage')
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
    signal_gen.run_backtest_on_all_pairs(
        top_n=args.top_n,
        lookback_days=args.lookback_days
    )

if __name__ == '__main__':
    main() 
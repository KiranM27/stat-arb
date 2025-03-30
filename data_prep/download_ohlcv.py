#!/usr/bin/env python
import ccxt
import pandas as pd
import time
import os
import argparse
import sys
from datetime import datetime, timedelta
from top_tokens import get_top_tokens

def download_ohlcv_data(days=730, skip_existing=True):
    # Get the project root directory (parent directory of the script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Create directory for data if it doesn't exist
    data_dir = os.path.join(project_root, 'ohlcv_data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Initialize exchange
    exchange = ccxt.binance()
    
    # Calculate timestamp for specified days ago
    now = datetime.now()
    start_date = now - timedelta(days=days)
    since = int(start_date.timestamp() * 1000)  # milliseconds
    
    # Get top 20 tokens
    top_tokens = get_top_tokens()
    
    for i, token in enumerate(top_tokens):
        symbol = token['symbol']
        market_symbol = f"{symbol}/USDT"
        filename = os.path.join(data_dir, f"{symbol}_USDT_daily.csv")
        
        # Skip if file already exists and skip_existing is True
        if os.path.exists(filename) and skip_existing:
            print(f"Skipping {market_symbol}, data file already exists.")
            continue
        
        try:
            print(f"Downloading data for {market_symbol} ({i+1}/20)...")
            
            # Fetch OHLCV data (daily candles for specified period)
            ohlcv = exchange.fetch_ohlcv(market_symbol, '1d', since=since, limit=1000)
            
            # Save raw data to CSV
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d'))
            df.to_csv(filename, index=False)
            
            print(f"Saved data to {filename}. {len(df)} days of data.")
            
            # Add delay to avoid rate limiting
            time.sleep(1)
        
        except Exception as e:
            print(f"Error downloading data for {market_symbol}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download OHLCV data for top cryptocurrencies')
    parser.add_argument('--days', type=int, default=730, help='Number of days of historical data to download (default: 730 days/2 years)')
    parser.add_argument('--force', action='store_true', help='Force download even if files already exist')
    args = parser.parse_args()
    
    time_period = "2 years" if args.days == 730 else f"{args.days} days"
    print(f"Starting download of {time_period} of daily OHLCV data for top 20 tokens...")
    download_ohlcv_data(days=args.days, skip_existing=not args.force)
    print("Download complete!") 
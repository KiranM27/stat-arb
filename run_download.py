#!/usr/bin/env python
import sys
import os

# Add data_prep to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_prep_dir = os.path.join(script_dir, 'data_prep')
sys.path.append(data_prep_dir)

# Import and run download
from data_prep.download_ohlcv import download_ohlcv_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download OHLCV data for top cryptocurrencies')
    parser.add_argument('--days', type=int, default=730, help='Number of days of historical data to download (default: 730 days/2 years)')
    parser.add_argument('--force', action='store_true', help='Force download even if files already exist')
    args = parser.parse_args()
    
    time_period = "2 years" if args.days == 730 else f"{args.days} days"
    print(f"Starting download of {time_period} of daily OHLCV data for top 20 tokens...")
    download_ohlcv_data(days=args.days, skip_existing=not args.force)
    print("Download complete!") 
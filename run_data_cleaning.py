#!/usr/bin/env python
import sys
import os

# Add data_cleaning to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_cleaning_dir = os.path.join(script_dir, 'data_cleaning')
sys.path.append(data_cleaning_dir)

# Import and run data cleaning
from data_cleaning.clean_ohlcv_data import clean_all_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean OHLCV data')
    parser.add_argument('--days', type=int, help='Expected number of days in each file')
    args = parser.parse_args()
    
    print("Starting data cleaning process...")
    clean_all_data(args.days) 
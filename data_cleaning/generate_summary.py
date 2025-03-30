#!/usr/bin/env python
import pandas as pd
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def generate_data_summary():
    """Generate a summary of the cleaned data"""
    cleaned_data_dir = 'cleaned_ohlcv_data'
    
    if not os.path.exists(cleaned_data_dir):
        print(f"Error: Cleaned data directory {cleaned_data_dir} does not exist!")
        return
    
    summary_data = []
    
    for file_name in os.listdir(cleaned_data_dir):
        if not file_name.endswith('_USDT_daily.csv'):
            continue
        
        symbol = file_name.split('_')[0]
        file_path = os.path.join(cleaned_data_dir, file_name)
        
        try:
            # Read the data
            df = pd.read_csv(file_path)
            
            # Convert timestamp to datetime if it's string
            if isinstance(df['timestamp'].iloc[0], str):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate statistics
            summary = {
                'Symbol': symbol,
                'Start Date': df['timestamp'].min(),
                'End Date': df['timestamp'].max(),
                'Days of Data': len(df),
                'Avg Daily Volume': df['volume'].mean(),
                'Avg Price': df['close'].mean(),
                'Min Price': df['low'].min(),
                'Max Price': df['high'].max(),
                'Last Price': df['close'].iloc[-1],
                'Price Change %': ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100 if len(df) > 1 else 0
            }
            
            summary_data.append(summary)
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    # Create summary DataFrame
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        summary_df.to_csv('data_summary.csv', index=False)
        print(f"Summary saved to data_summary.csv")
        
        # Display statistics
        print("\nData Summary Statistics:")
        print(f"Total tokens: {len(summary_df)}")
        print(f"Average days of data: {summary_df['Days of Data'].mean():.1f}")
        print(f"Maximum price change %: {summary_df['Symbol'][summary_df['Price Change %'].idxmax()]} ({summary_df['Price Change %'].max():.2f}%)")
        print(f"Minimum price change %: {summary_df['Symbol'][summary_df['Price Change %'].idxmin()]} ({summary_df['Price Change %'].min():.2f}%)")
    else:
        print("No data found to summarize")

if __name__ == '__main__':
    generate_data_summary() 
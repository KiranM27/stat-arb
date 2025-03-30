#!/usr/bin/env python
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import itertools

def load_all_data():
    """Load all cleaned data and calculate returns"""
    cleaned_data_dir = 'cleaned_ohlcv_data'
    
    if not os.path.exists(cleaned_data_dir):
        print(f"Error: Cleaned data directory {cleaned_data_dir} does not exist!")
        return None
    
    # Dictionary to store returns dataframes
    returns_data = {}
    
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
            
            # Sort by timestamp to ensure correct return calculation
            df = df.sort_values('timestamp')
            
            # Calculate daily returns based on close prices
            df['return'] = df['close'].pct_change()
            
            # Remove the first row (NaN return)
            df = df.dropna(subset=['return'])
            
            # Only include tokens with sufficient data points (at least 5 days)
            if len(df) >= 5:
                # Store in dictionary with timestamp as index
                returns_data[symbol] = df.set_index('timestamp')['return']
                print(f"Loaded {symbol} data: {len(df)} days")
            else:
                print(f"Skipping {symbol}: insufficient data ({len(df)} days)")
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    return returns_data

def create_returns_matrix(returns_data):
    """Create a dataframe with all returns aligned by timestamp"""
    # Create a list of Series
    series_list = []
    symbols = []
    
    for symbol, returns in returns_data.items():
        series_list.append(returns)
        symbols.append(symbol)
    
    # Concatenate all series into a single dataframe
    if series_list:
        returns_df = pd.concat(series_list, axis=1)
        returns_df.columns = symbols
        
        # Note: We're not dropping NaN values globally anymore
        # This allows for pairwise correlations even with missing data
        print(returns_df.head())
        return returns_df
    
    return None

def compute_correlations(returns_df, threshold=0.5, min_periods=100):
    """Compute pairwise correlations and filter by threshold"""
    if returns_df is None or returns_df.empty:
        print("Error: No returns data available.")
        return None, None
    
    # Compute correlation matrix with min_periods to handle missing data
    # min_periods is the minimum number of overlapping non-NA values required
    corr_matrix = returns_df.corr(min_periods=min_periods)
    
    # Create directory for outputs if it doesn't exist
    output_dir = 'correlation_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save correlation matrix to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    corr_matrix.to_csv(f"{output_dir}/correlation_matrix_{timestamp}.csv")
    
    # Plot and save heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title(f'Cryptocurrency Return Correlations (Daily)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap_{timestamp}.png", dpi=300)
    
    # Find pairs with correlation above threshold
    high_corr_pairs = []
    symbols = list(corr_matrix.columns)
    
    for i, j in itertools.combinations(range(len(symbols)), 2):
        symbol1 = symbols[i]
        symbol2 = symbols[j]
        correlation = corr_matrix.iloc[i, j]
        
        # Skip NaN correlations (can happen if min_periods not met)
        if pd.isna(correlation):
            continue
            
        if abs(correlation) >= threshold:
            # Count overlapping non-NA periods
            overlap_days = (~np.isnan(returns_df[symbol1]) & ~np.isnan(returns_df[symbol2])).sum()
            
            high_corr_pairs.append({
                'Symbol1': symbol1, 
                'Symbol2': symbol2, 
                'Correlation': correlation,
                'Overlap_Days': overlap_days
            })
    
    # Convert to dataframe and sort by absolute correlation (descending)
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        high_corr_df['Abs_Correlation'] = high_corr_df['Correlation'].abs()
        high_corr_df = high_corr_df.sort_values('Abs_Correlation', ascending=False)
        
        # Save high correlation pairs to CSV
        high_corr_df.to_csv(f"{output_dir}/high_correlation_pairs_{timestamp}.csv", index=False)
        
        return high_corr_df, corr_matrix
    
    return None, corr_matrix

def analyze_returns_correlation(correlation_threshold=0.5, min_periods=5):
    """Perform correlation analysis on cryptocurrency returns"""
    print("Loading data and calculating returns...")
    returns_data = load_all_data()
    
    if not returns_data:
        print("No data available for analysis.")
        return
    
    print("Creating aligned returns matrix...")
    returns_matrix = create_returns_matrix(returns_data)
    
    print(f"Computing correlations with threshold {correlation_threshold}...")
    high_corr_pairs, corr_matrix = compute_correlations(returns_matrix, correlation_threshold, min_periods)
    
    if high_corr_pairs is not None and not high_corr_pairs.empty:
        print("\nTop 10 Highly Correlated Pairs:")
        display_cols = ['Symbol1', 'Symbol2', 'Correlation', 'Overlap_Days']
        print(high_corr_pairs[display_cols].head(10))
        print(f"\nFound {len(high_corr_pairs)} pairs with correlation >= {correlation_threshold}")
        
        # Also print negative correlations (potentially useful for hedging)
        neg_corr = high_corr_pairs[high_corr_pairs['Correlation'] < 0]
        if not neg_corr.empty:
            print("\nTop 5 Negatively Correlated Pairs (potential hedging opportunities):")
            print(neg_corr[display_cols].head(5))
    else:
        print(f"No pairs found with correlation >= {correlation_threshold}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Perform correlation analysis on cryptocurrency returns')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Correlation threshold (default: 0.5)')
    parser.add_argument('--min-periods', type=int, default=5,
                        help='Minimum number of overlapping periods required (default: 5)')
    args = parser.parse_args()
    
    analyze_returns_correlation(args.threshold, args.min_periods) 
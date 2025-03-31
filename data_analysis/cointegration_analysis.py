#!/usr/bin/env python
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

def load_price_data():
    """Load price data for all tokens"""
    cleaned_data_dir = 'cleaned_ohlcv_data'
    
    if not os.path.exists(cleaned_data_dir):
        print(f"Error: Cleaned data directory {cleaned_data_dir} does not exist!")
        return None
    
    # Dictionary to store price dataframes
    price_data = {}
    
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
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Only include tokens with sufficient data points (at least 60 days for robust analysis)
            if len(df) >= 60:
                # Store in dictionary with timestamp as index
                price_data[symbol] = df.set_index('timestamp')['close']
                print(f"Loaded {symbol} price data: {len(df)} days")
            else:
                print(f"Skipping {symbol}: insufficient data ({len(df)} days)")
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    return price_data

def load_correlation_pairs(threshold=0.5, min_overlap=100):
    """Load high correlation pairs from the most recent correlation analysis results"""
    corr_dir = 'correlation_results'
    
    if not os.path.exists(corr_dir):
        print(f"Error: Correlation results directory {corr_dir} does not exist!")
        return None
    
    # Find the most recent correlation pairs file
    corr_files = [f for f in os.listdir(corr_dir) if f.startswith('high_correlation_pairs_')]
    if not corr_files:
        print("No correlation results found.")
        return None
    
    latest_file = sorted(corr_files)[-1]
    file_path = os.path.join(corr_dir, latest_file)
    
    # Load the correlation pairs
    corr_pairs = pd.read_csv(file_path)
    
    # Filter by threshold and minimum overlap
    filtered_pairs = corr_pairs[
        (corr_pairs['Abs_Correlation'] >= threshold) & 
        (corr_pairs['Overlap_Days'] >= min_overlap)
    ]
    
    return filtered_pairs

def test_cointegration(price1, price2, max_lag=None, method='adf', alpha=0.05):
    """
    Test if two price series are cointegrated
    
    Parameters:
    price1, price2: Price series
    max_lag: Maximum lag for ADF test, None for automatic selection
    method: 'adf' for Augmented Dickey-Fuller or 'coint' for Engle-Granger
    alpha: Significance level
    
    Returns:
    is_cointegrated: Boolean indicating if pair is cointegrated
    p_value: P-value of the test
    """
    # Ensure indexes are aligned
    common_idx = price1.index.intersection(price2.index)
    if len(common_idx) < 60:  # Minimum data points for reliable test
        return False, 1.0, None, None
    
    p1 = price1.loc[common_idx]
    p2 = price2.loc[common_idx]
    
    if method == 'adf':
        # Calculate the spread
        # First, find the hedge ratio using linear regression
        X = sm.add_constant(p1.values.reshape(-1, 1))
        model = sm.OLS(p2.values, X).fit()
        beta = model.params[1]  # Hedge ratio
        alpha_const = model.params[0]  # Intercept
        
        # Calculate the spread/residuals
        spread = p2 - (alpha_const + beta * p1)
        
        # Run ADF test on the spread
        result = adfuller(spread, maxlag=max_lag)
        p_value = result[1]
        
        # Lower p-value means we can reject the null hypothesis of non-stationarity
        is_cointegrated = p_value <= alpha
        
        return is_cointegrated, p_value, beta, alpha_const
    
    elif method == 'coint':
        # Use Engle-Granger test
        result = coint(p1, p2, maxlag=max_lag)
        p_value = result[1]
        
        # Calculate the hedge ratio and intercept separately
        X = sm.add_constant(p1.values.reshape(-1, 1))
        model = sm.OLS(p2.values, X).fit()
        beta = model.params[1]  # Hedge ratio
        alpha_const = model.params[0]  # Intercept
        
        is_cointegrated = p_value <= alpha
        
        return is_cointegrated, p_value, beta, alpha_const
    
    return False, 1.0, None, None

def calculate_half_life(spread, max_lag=20):
    """Calculate the half-life of mean reversion for a spread series"""
    # Lag the spread
    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag
    
    # Remove NAs and align the data
    df = pd.DataFrame({'spread_lag': spread_lag, 'spread_diff': spread_diff})
    df = df.dropna()
    
    # If not enough data points, return infinity
    if len(df) < 10:
        return np.inf
    
    # Run regression: spread_diff = gamma * spread_lag + error
    X = df['spread_lag'].values.reshape(-1, 1)
    y = df['spread_diff'].values
    
    model = LinearRegression().fit(X, y)
    gamma = model.coef_[0]
    
    # Calculate half-life: ln(2) / gamma if gamma is negative, else inf
    if gamma < 0:
        half_life = -np.log(2) / gamma
        return half_life
    else:
        return np.inf  # Not mean-reverting

def analyze_cointegrated_pairs(price_data, corr_pairs, method='adf', alpha=0.05, 
                               min_half_life=1, max_half_life=30, 
                               lookback_periods=None):
    """
    Analyze pairs for cointegration and compute hedge ratios
    
    Parameters:
    price_data: Dictionary of price time series
    corr_pairs: DataFrame of correlated pairs
    method: Cointegration test method ('adf' or 'coint')
    alpha: Significance level for cointegration test
    min_half_life, max_half_life: Range of acceptable half-life values
    lookback_periods: List of lookback periods to test (in days)
    
    Returns:
    DataFrame of cointegrated pairs with stats
    """
    if lookback_periods is None:
        lookback_periods = [None]  # Use all available data
    
    results = []
    
    for _, row in corr_pairs.iterrows():
        symbol1 = row['Symbol1']
        symbol2 = row['Symbol2']
        
        if symbol1 not in price_data or symbol2 not in price_data:
            print(f"Skipping {symbol1}-{symbol2}: Price data not available")
            continue
        
        price1 = price_data[symbol1]
        price2 = price_data[symbol2]
        
        # Test cointegration over different lookback periods
        cointegrated_periods = 0
        latest_result = None
        
        for period in lookback_periods:
            if period is not None:
                # Use only the last 'period' days
                p1 = price1.iloc[-period:] if len(price1) >= period else price1
                p2 = price2.iloc[-period:] if len(price2) >= period else price2
            else:
                # Use all available data
                p1 = price1
                p2 = price2
            
            is_coint, p_value, beta, alpha_const = test_cointegration(
                p1, p2, method=method, alpha=alpha
            )
            
            if period is None or period == lookback_periods[-1]:
                latest_result = (is_coint, p_value, beta, alpha_const, p1, p2)
            
            if is_coint:
                cointegrated_periods += 1
        
        # Detailed analysis on the latest period
        if latest_result and latest_result[0]:  # If cointegrated in the latest period
            is_coint, p_value, beta, alpha_const, p1, p2 = latest_result
            
            # Calculate the spread
            spread = p2 - (alpha_const + beta * p1)
            
            # Calculate half-life of mean reversion
            half_life = calculate_half_life(spread)
            
            # Calculate additional metrics
            spread_mean = spread.mean()
            spread_std = spread.std()
            
            # Calculate z-score of current spread
            current_spread = spread.iloc[-1]
            z_score = (current_spread - spread_mean) / spread_std
            
            # Filter by half-life
            if min_half_life <= half_life <= max_half_life:
                results.append({
                    'Symbol1': symbol1,
                    'Symbol2': symbol2,
                    'Correlation': row['Correlation'],
                    'Cointegrated_Periods': cointegrated_periods,
                    'Total_Periods': len(lookback_periods),
                    'Cointegration_Score': cointegrated_periods / len(lookback_periods),
                    'Latest_P_Value': p_value,
                    'Hedge_Ratio': beta,
                    'Intercept': alpha_const,
                    'Half_Life': half_life,
                    'Spread_Mean': spread_mean,
                    'Spread_Std': spread_std,
                    'Current_Spread': current_spread,
                    'Z_Score': z_score,
                    'Data_Points': len(spread)
                })
    
    if not results:
        print("No cointegrated pairs found with the specified criteria.")
        return None
    
    # Convert to DataFrame and sort by cointegration score and correlation
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['Cointegration_Score', 'Correlation'], ascending=False)
    
    return results_df

def plot_cointegrated_pair(price_data, symbol1, symbol2, hedge_ratio, intercept, 
                           lookback=None, save_dir='pair_plots'):
    """Plot a cointegrated pair and its spread"""
    # Create directory for plots if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    price1 = price_data[symbol1]
    price2 = price_data[symbol2]
    
    # Use only lookback period if specified
    if lookback is not None:
        price1 = price1.iloc[-lookback:] if len(price1) >= lookback else price1
        price2 = price2.iloc[-lookback:] if len(price2) >= lookback else price2
    
    # Align indexes
    common_idx = price1.index.intersection(price2.index)
    price1 = price1.loc[common_idx]
    price2 = price2.loc[common_idx]
    
    # Calculate the spread
    spread = price2 - (intercept + hedge_ratio * price1)
    
    # Calculate z-score
    spread_mean = spread.mean()
    spread_std = spread.std()
    z_score = (spread - spread_mean) / spread_std
    
    # Create a 2x1 subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot normalized prices
    norm_price1 = price1 / price1.iloc[0]
    norm_price2 = price2 / price2.iloc[0]
    
    ax1.plot(norm_price1.index, norm_price1, label=f'{symbol1} (normalized)')
    ax1.plot(norm_price2.index, norm_price2, label=f'{symbol2} (normalized)')
    ax1.set_title(f'Normalized Prices: {symbol1} vs {symbol2}')
    ax1.legend()
    ax1.grid(True)
    
    # Plot z-score of spread
    ax2.plot(z_score.index, z_score)
    ax2.axhline(y=0, color='r', linestyle='-')
    ax2.axhline(y=1, color='g', linestyle='--')
    ax2.axhline(y=-1, color='g', linestyle='--')
    ax2.axhline(y=2, color='y', linestyle='--')
    ax2.axhline(y=-2, color='y', linestyle='--')
    ax2.set_title(f'Z-Score of Spread (Hedge Ratio: {hedge_ratio:.4f})')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{symbol1}_{symbol2}_pair.png", dpi=300)
    plt.close()

def save_candidate_pairs(results_df, top_n=10):
    """Save candidate pairs to CSV and plot top pairs"""
    if results_df is None or results_df.empty:
        return None
    
    # Create directory for outputs
    output_dir = 'cointegration_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save full results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_csv(f"{output_dir}/cointegrated_pairs_{timestamp}.csv", index=False)
    
    # Save top N candidates
    top_candidates = results_df.head(top_n)
    top_candidates.to_csv(f"{output_dir}/top_cointegrated_pairs_{timestamp}.csv", index=False)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")
    print(f"\nTop {top_n} Cointegrated Pairs:")
    
    display_cols = [
        'Symbol1', 'Symbol2', 'Correlation', 'Cointegration_Score',
        'Latest_P_Value', 'Hedge_Ratio', 'Half_Life', 'Z_Score'
    ]
    
    if top_n > 0:
        print(top_candidates[display_cols])
    
    return top_candidates

def run_cointegration_analysis(corr_threshold=0.5, min_overlap=100, 
                               coint_method='adf', alpha=0.05,
                               min_half_life=1, max_half_life=30,
                               lookback_days=None, top_n=10):
    """
    Run full cointegration analysis pipeline
    
    Parameters:
    corr_threshold: Minimum correlation threshold
    min_overlap: Minimum overlapping days for correlation
    coint_method: Cointegration test method ('adf' or 'coint')
    alpha: Significance level for cointegration test
    min_half_life, max_half_life: Range of acceptable half-life values
    lookback_days: List of lookback periods to test (in days)
    top_n: Number of top pairs to display
    """
    # Default lookback periods if not specified
    if lookback_days is None:
        lookback_days = [None, 365, 180, 90]  # None means all available data
    
    print("Loading price data...")
    price_data = load_price_data()
    
    if not price_data:
        print("No price data available for analysis.")
        return
    
    print(f"Loading correlation pairs (threshold: {corr_threshold}, min overlap: {min_overlap})...")
    corr_pairs = load_correlation_pairs(threshold=corr_threshold, min_overlap=min_overlap)
    
    if corr_pairs is None or corr_pairs.empty:
        print("No correlation pairs available for analysis.")
        return
    
    print(f"Found {len(corr_pairs)} correlated pairs to analyze.")
    
    print(f"Testing cointegration (method: {coint_method}, alpha: {alpha})...")
    results = analyze_cointegrated_pairs(
        price_data, corr_pairs, method=coint_method, alpha=alpha,
        min_half_life=min_half_life, max_half_life=max_half_life,
        lookback_periods=lookback_days
    )
    
    if results is not None:
        print(f"Found {len(results)} cointegrated pairs.")
        
        # Save results and get top candidates
        top_candidates = save_candidate_pairs(results, top_n=top_n)
        
        # Plot top candidates
        if top_candidates is not None:
            print("Generating plots for top pairs...")
            for _, row in top_candidates.iterrows():
                plot_cointegrated_pair(
                    price_data, 
                    row['Symbol1'], 
                    row['Symbol2'], 
                    row['Hedge_Ratio'],
                    row['Intercept']
                )
            print("Plots generated in 'pair_plots' directory.")
    else:
        print("No cointegrated pairs found.")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Perform cointegration analysis on correlated cryptocurrency pairs')
    parser.add_argument('--corr-threshold', type=float, default=0.5, 
                        help='Correlation threshold (default: 0.5)')
    parser.add_argument('--min-overlap', type=int, default=100,
                        help='Minimum overlapping days for correlation (default: 100)')
    parser.add_argument('--coint-method', type=str, default='adf', choices=['adf', 'coint'],
                        help='Cointegration test method (default: adf)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level for cointegration test (default: 0.05)')
    parser.add_argument('--min-half-life', type=float, default=1,
                        help='Minimum half-life in days (default: 1)')
    parser.add_argument('--max-half-life', type=float, default=30,
                        help='Maximum half-life in days (default: 30)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top pairs to display (default: 10)')
    args = parser.parse_args()
    
    # Define lookback periods (in days)
    lookback_days = [None, 365, 180, 90]  # None means all available data
    
    run_cointegration_analysis(
        corr_threshold=args.corr_threshold,
        min_overlap=args.min_overlap,
        coint_method=args.coint_method,
        alpha=args.alpha,
        min_half_life=args.min_half_life,
        max_half_life=args.max_half_life,
        lookback_days=lookback_days,
        top_n=args.top_n
    ) 
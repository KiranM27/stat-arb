#!/usr/bin/env python
import sys
import os

# Add data_analysis to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_analysis_dir = os.path.join(script_dir, 'data_analysis')
sys.path.append(data_analysis_dir)

# Import and run cointegration analysis
from data_analysis.cointegration_analysis import run_cointegration_analysis

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Perform cointegration analysis on cryptocurrency pairs')
    parser.add_argument('--corr-threshold', type=float, default=0.5, 
                        help='Correlation threshold (default: 0.5)')
    parser.add_argument('--min-overlap', type=int, default=100,
                        help='Minimum overlapping days for correlation (default: 100)')
    parser.add_argument('--coint-method', type=str, default='adf', choices=['adf', 'coint'],
                        help='Cointegration test method (default: adf)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level for cointegration test (default: 0.05)')
    parser.add_argument('--min-half-life', type=float, default=1.0,
                        help='Minimum half-life in days (default: 1.0)')
    parser.add_argument('--max-half-life', type=float, default=30.0,
                        help='Maximum half-life in days (default: 30.0)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top pairs to display (default: 10)')
    args = parser.parse_args()
    
    # Define lookback periods (in days)
    lookback_days = [None, 365, 180, 90]  # None means all available data
    
    print("Starting cointegration analysis of cryptocurrency pairs...")
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
#!/usr/bin/env python
import sys
import os

# Add data_analysis to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_analysis_dir = os.path.join(script_dir, 'data_analysis')
sys.path.append(data_analysis_dir)

# Import and run correlation analysis
from data_analysis.correlation_analysis import analyze_returns_correlation

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Perform correlation analysis on cryptocurrency returns')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Correlation threshold (default: 0.5)')
    parser.add_argument('--min-periods', type=int, default=5,
                        help='Minimum number of overlapping periods required (default: 5)')
    args = parser.parse_args()
    
    print("Starting correlation analysis of crypto returns...")
    analyze_returns_correlation(args.threshold, args.min_periods) 
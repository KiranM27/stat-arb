#!/usr/bin/env python
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='data_cleaning_log.txt'
)

def create_directories():
    """Create necessary directories if they don't exist"""
    # Original data directory
    original_data_dir = 'ohlcv_data'
    if not os.path.exists(original_data_dir):
        logging.error(f"Original data directory {original_data_dir} does not exist!")
        return False
        
    # Cleaned data directory
    cleaned_data_dir = 'cleaned_ohlcv_data'
    if not os.path.exists(cleaned_data_dir):
        os.makedirs(cleaned_data_dir)
        logging.info(f"Created directory for cleaned data: {cleaned_data_dir}")
    
    return original_data_dir, cleaned_data_dir

def verify_date_range(df, symbol, expected_days=None):
    """Check if the data covers the expected time range"""
    if df.empty:
        logging.warning(f"{symbol}: DataFrame is empty!")
        return False
    
    # Convert timestamp to datetime if it's string
    if isinstance(df['timestamp'].iloc[0], str):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    actual_days = (df['timestamp'].max() - df['timestamp'].min()).days + 1
    
    if expected_days and actual_days < expected_days * 0.9:  # Allow for 10% fewer days
        logging.warning(f"{symbol}: Expected {expected_days} days but found {actual_days} days")
        return False
    
    # Check for gaps in the date range
    all_dates = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='D')
    missing_dates = set(all_dates) - set(df['timestamp'])
    
    if missing_dates:
        logging.warning(f"{symbol}: Missing {len(missing_dates)} dates in the expected range")
        logging.debug(f"{symbol}: Missing dates: {missing_dates}")
    
    return True

def check_data_types(df, symbol):
    """Verify column formats and convert to appropriate data types"""
    # Expected columns and types
    expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # Check if all expected columns exist
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        logging.error(f"{symbol}: Missing columns: {missing_columns}")
        return False
    
    # Convert timestamp to datetime if it's a string
    if df['timestamp'].dtype == 'object':
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            logging.error(f"{symbol}: Failed to convert timestamp to datetime: {e}")
            return False
    
    # Convert numeric columns
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception as e:
            logging.error(f"{symbol}: Failed to convert {col} to numeric: {e}")
            return False
    
    return True

def handle_missing_values(df, symbol):
    """Handle missing or erroneous rows"""
    # Count initial rows
    initial_rows = len(df)
    
    # Check for NaN values in critical columns
    critical_cols = ['open', 'high', 'low', 'close']
    missing_critical = df[critical_cols].isna().any(axis=1)
    
    if missing_critical.any():
        logging.warning(f"{symbol}: Found {missing_critical.sum()} rows with missing critical values")
        df = df[~missing_critical]
    
    # Check for duplicate timestamps
    duplicates = df['timestamp'].duplicated()
    if duplicates.any():
        logging.warning(f"{symbol}: Found {duplicates.sum()} duplicate timestamps")
        df = df.drop_duplicates(subset=['timestamp'])
    
    # Report rows removed
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        logging.info(f"{symbol}: Removed {rows_removed} problematic rows")
    
    return df

def filter_suspicious_records(df, symbol):
    """Filter out suspicious records like zero volume or extreme price outliers"""
    # Count initial rows
    initial_rows = len(df)
    
    # Filter out zero or near-zero volume records
    min_volume_threshold = df['volume'].quantile(0.01)  # Bottom 1% as threshold
    zero_volume = df['volume'] <= min_volume_threshold
    
    if zero_volume.any():
        logging.warning(f"{symbol}: Found {zero_volume.sum()} rows with zero/near-zero volume")
        df = df[~zero_volume]
    
    # Detect price outliers using IQR method
    for col in ['open', 'high', 'low', 'close']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        if outliers.any():
            logging.warning(f"{symbol}: Found {outliers.sum()} price outliers in {col}")
            
            # Cap outliers instead of removing
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound
    
    # Report on changes
    rows_affected = initial_rows - len(df)
    if rows_affected > 0:
        logging.info(f"{symbol}: Filtered {rows_affected} suspicious records")
    
    return df

def ensure_ohlc_integrity(df, symbol):
    """Ensure OHLC integrity (high ≥ open, close, low; low ≤ open, close)"""
    # Check high >= open, close, low
    high_errors = (df['high'] < df['open']) | (df['high'] < df['close']) | (df['high'] < df['low'])
    
    # Check low <= open, close
    low_errors = (df['low'] > df['open']) | (df['low'] > df['close'])
    
    if high_errors.any() or low_errors.any():
        logging.warning(f"{symbol}: Found {high_errors.sum() + low_errors.sum()} OHLC integrity issues")
        
        # Fix high values
        df.loc[high_errors, 'high'] = df.loc[high_errors, ['open', 'close', 'low']].max(axis=1)
        
        # Fix low values
        df.loc[low_errors, 'low'] = df.loc[low_errors, ['open', 'close']].min(axis=1)
    
    return df

def clean_data_file(file_path, output_dir, expected_days=None):
    """Clean a single data file"""
    file_name = os.path.basename(file_path)
    symbol = file_name.split('_')[0]
    
    logging.info(f"Cleaning data for {symbol}...")
    
    try:
        # Read the data
        df = pd.read_csv(file_path)
        
        # Apply cleaning functions
        verify_date_range(df, symbol, expected_days)
        valid_types = check_data_types(df, symbol)
        
        if not valid_types:
            logging.error(f"{symbol}: Data type issues could not be resolved. Skipping file.")
            return False
        
        df = handle_missing_values(df, symbol)
        df = filter_suspicious_records(df, symbol)
        df = ensure_ohlc_integrity(df, symbol)
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Save cleaned data
        output_path = os.path.join(output_dir, file_name)
        df.to_csv(output_path, index=False)
        
        logging.info(f"{symbol}: Successfully cleaned and saved to {output_path}")
        return True
    
    except Exception as e:
        logging.error(f"{symbol}: Error cleaning data: {e}")
        return False

def clean_all_data(days=None):
    """Clean all data files in the ohlcv_data directory"""
    original_dir, cleaned_dir = create_directories()
    if not original_dir:
        return
    
    success_count = 0
    fail_count = 0
    
    for file_name in os.listdir(original_dir):
        if not file_name.endswith('_USDT_daily.csv'):
            continue
        
        file_path = os.path.join(original_dir, file_name)
        
        if clean_data_file(file_path, cleaned_dir, days):
            success_count += 1
        else:
            fail_count += 1
    
    logging.info(f"Cleaning completed: {success_count} files processed successfully, {fail_count} files failed")
    print(f"Cleaning completed: {success_count} files processed successfully, {fail_count} files failed")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean OHLCV data')
    parser.add_argument('--days', type=int, help='Expected number of days in each file')
    args = parser.parse_args()
    
    clean_all_data(args.days) 
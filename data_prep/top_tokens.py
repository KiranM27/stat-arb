#!/usr/bin/env python
import ccxt
import pandas as pd
from datetime import datetime

def get_top_tokens():
    # Initialize the exchange
    exchange = ccxt.binance()
    
    # Fetch all markets
    exchange.load_markets()
    
    # Get all tickers
    tickers = exchange.fetch_tickers()
    
    # Create a list to store token data
    token_data = []
    
    # Process tickers that are trading against USDT
    for symbol, ticker in tickers.items():
        if '/USDT' in symbol:
            base_currency = symbol.split('/')[0]
            
            # Skip stablecoins
            if base_currency in ['USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'FDUSD']:
                continue
                
            if 'quoteVolume' in ticker and ticker['quoteVolume'] is not None:
                token_data.append({
                    'symbol': base_currency,
                    'price': ticker['last'],
                    'volume_24h': ticker['quoteVolume'],
                    'market_cap': None  # Binance doesn't provide market cap directly
                })
    
    # Sort by 24h volume as a proxy for market cap
    token_data.sort(key=lambda x: x['volume_24h'] if x['volume_24h'] else 0, reverse=True)
    
    # Return top 20
    return token_data[:20]

if __name__ == '__main__':
    print(f"Fetching top 20 tokens by trading volume (proxy for market cap) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tokens = get_top_tokens()
    
    # Create a DataFrame for better display
    df = pd.DataFrame(tokens)
    
    # Format the output
    pd.set_option('display.float_format', '${:.2f}'.format)
    print(df[['symbol', 'price', 'volume_24h']])
    print("\nNote: Using 24h trading volume as a proxy for market cap as CCXT doesn't provide direct market cap data") 
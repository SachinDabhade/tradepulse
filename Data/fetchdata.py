import sqlite3
from Config.config import config
import yfinance as yf
from datetime import datetime, timedelta, date
import pandas as pd
import json
import os
from Config.metadata import get_stock_metadata, load_existing_metadata, update_metadata_file
from typing import Optional, Tuple, List, Union

# # Metadata file path
# metadata_file_path = "./Config/stock_metadata.json"

# # def check_metadata_for_symbol(symbol: str, interval: str = "1d", start_date: Optional[date] = None, end_date: Optional[date] = None) -> Tuple[bool, Optional[dict]]:
# #     """
# #     Check if data exists in metadata and if it covers the required date range.
# #     Returns (is_fully_available, metadata_dict)
# #     """
# #     metadata = load_existing_metadata(metadata_file_path)
# #     key = f"{symbol}_{interval}"
    
# #     if key not in metadata:
# #         return False, None
    
# #     symbol_metadata = metadata[key]
# #     meta_start = datetime.strptime(symbol_metadata["start_date"], "%Y-%m-%d").date()
# #     meta_end = datetime.strptime(symbol_metadata["end_date"], "%Y-%m-%d").date()
    
# #     # If no specific date range requested, check if we have any data
# #     if start_date is None and end_date is None:
# #         return True, symbol_metadata
    
# #     # Check if metadata covers the required date range
# #     if start_date and meta_start > start_date:
# #         return False, symbol_metadata
# #     if end_date and meta_end < end_date:
# #         return False, symbol_metadata
    
# #     return True, symbol_metadata

# def get_missing_date_ranges(symbol: str, interval: str = "1d", start_date: Optional[date] = None, end_date: Optional[date] = None) -> List[Tuple[Optional[date], Optional[date]]]:
#     """
#     Determine what date ranges need to be downloaded for a symbol.
#     Returns list of (start_date, end_date) tuples for missing ranges.
#     """
#     metadata = load_existing_metadata(metadata_file_path)
#     key = f"{symbol}_{interval}"
    
#     if key not in metadata:
#         # No data exists, need to download full range
#         return [(start_date, end_date)]
    
#     symbol_metadata = metadata[key]
#     meta_start = datetime.strptime(symbol_metadata["start_date"], "%Y-%m-%d").date()
#     meta_end = datetime.strptime(symbol_metadata["end_date"], "%Y-%m-%d").date()
    
#     missing_ranges = []
    
#     # If no specific dates requested, check if we need to update to today
#     if start_date is None and end_date is None:
#         today = datetime.now().date()
#         if meta_end < today:
#             missing_ranges.append((meta_end + timedelta(days=1), today))
#         return missing_ranges
    
#     # Check for missing data before existing range
#     if start_date and meta_start > start_date:
#         missing_ranges.append((start_date, meta_start - timedelta(days=1)))
    
#     # Check for missing data after existing range
#     if end_date and meta_end < end_date:
#         missing_ranges.append((meta_end + timedelta(days=1), end_date))
    
#     return missing_ranges

# def update_metadata_for_symbol(symbol: str, interval: str = "1d") -> bool:
#     """
#     Update metadata for a specific symbol.
#     """
#     metadata = get_stock_metadata(symbol, interval)
#     if metadata:
#         all_metadata = load_existing_metadata(metadata_file_path)
#         key = f"{symbol}_{interval}"
#         all_metadata[key] = metadata
        
#         os.makedirs(os.path.dirname(metadata_file_path), exist_ok=True)
#         with open(metadata_file_path, 'w') as f:
#             json.dump(all_metadata, f, indent=2)
#         return True
#     return False

# def fetch_and_store_multiple_stocks(ticker_list: List[str], start_date: Optional[date] = None, end_date: Optional[date] = None, db_path: str = config['PATHS']['STOCK_DATABASE'], interval: str = '1d'):
#     """
#     Fetch and store stock data with intelligent partial downloading.
#     """
#     print(f"Starting fetch_and_store_multiple_stocks for {len(ticker_list)} tickers...")
    
#     # Connect to database
#     try:
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()
#         print("Database connection established successfully.")
#     except Exception as e:
#         print(f"Failed to connect to database: {e}")
#         return

#     # Create the table if it doesn't exist
#     try:
#         cursor.execute(f"""
#             CREATE TABLE IF NOT EXISTS {config['DATABASE'][interval]} (
#                 ticker TEXT,
#                 date DATE,
#                 open REAL,
#                 high REAL,
#                 low REAL,
#                 close REAL,
#                 volume INTEGER,
#                 PRIMARY KEY (ticker, date)
#             )
#         """)
#         conn.commit()
#         print(f"Table {config['DATABASE'][interval]} ready.")
#     except Exception as e:
#         print(f"Failed to create table: {e}")
#         conn.close()
#         return

#     # Analyze what data needs to be downloaded for each ticker
#     print("Analyzing missing data ranges...")
#     tickers_to_process = []
#     for i, ticker in enumerate(ticker_list):
#         try:
#             print(f"Checking {ticker} ({i+1}/{len(ticker_list)})...")
#             missing_ranges = get_missing_date_ranges(ticker, interval, start_date, end_date)
#             if missing_ranges:
#                 tickers_to_process.append((ticker, missing_ranges))
#                 print(f"  {ticker}: {len(missing_ranges)} missing range(s)")
#             else:
#                 print(f"  {ticker}: No missing data")
#         except Exception as e:
#             print(f"  Error checking {ticker}: {e}")
#             continue
    
#     if not tickers_to_process:
#         print("All tickers already have complete data coverage. Skipping download.")
#         conn.close()
#         return

#     print(f"Processing {len(tickers_to_process)} tickers with missing data ranges...")

#     # Process each ticker with its missing ranges
#     for ticker_idx, (ticker, missing_ranges) in enumerate(tickers_to_process):
#         print(f"\nProcessing {ticker} ({ticker_idx+1}/{len(tickers_to_process)}) with {len(missing_ranges)} missing range(s)...")
        
#         for range_idx, (range_start, range_end) in enumerate(missing_ranges):
#             print(f"  Downloading range {range_idx+1}/{len(missing_ranges)}: {range_start} to {range_end}")
            
#             try:
#                 # Download specific date range with timeout
#                 import time
#                 start_time = time.time()
                
#                 stock_data = yf.download(
#                     ticker, 
#                     start=range_start, 
#                     end=range_end, 
#                     group_by='ticker', 
#                     auto_adjust=False, 
#                     threads=True, 
#                     interval=interval,
#                     progress=False  # Disable progress bar to avoid hanging
#                 )
                
#                 download_time = time.time() - start_time
#                 print(f"    Download completed in {download_time:.2f} seconds")
                
#                 if stock_data is None or stock_data.empty:
#                     print(f"    No data available for {ticker} in range {range_start} to {range_end}")
#                     continue
                
#                 print(f"    Downloaded {len(stock_data)} rows")
                
#                 # Process and insert data
#                 rows_to_insert = []
#                 # Ensure we're working with a DataFrame and it has a DatetimeIndex
#                 if isinstance(stock_data.index, pd.DatetimeIndex):
#                     stock_data = stock_data.dropna(subset=['Open', 'High', 'Low', 'Close'])
#                     print(f"    After dropping NaN: {len(stock_data)} rows")
                    
#                     for index, row in stock_data.iterrows():
#                         # Convert index to string date format
#                         date_str = str(index)[:10]  # Take first 10 characters (YYYY-MM-DD)
#                         rows_to_insert.append((
#                             ticker,
#                             date_str,
#                             row['Open'],
#                             row['High'],
#                             row['Low'],
#                             row['Close'],
#                             row['Volume']
#                         ))
                    
#                     if rows_to_insert:
#                         cursor.executemany(f"""
#                             INSERT OR IGNORE INTO {config['DATABASE'][interval]} (ticker, date, open, high, low, close, volume)
#                             VALUES (?, ?, ?, ?, ?, ?, ?)
#                         """, rows_to_insert)
#                         print(f"    Inserted {len(rows_to_insert)} rows for {ticker}")
#                     else:
#                         print(f"    No valid rows to insert for {ticker}")
#                 else:
#                     print(f"    Warning: stock_data index is not DatetimeIndex for {ticker}")
                
#             except Exception as e:
#                 print(f"    Failed to download data for {ticker} in range {range_start} to {range_end}: {e}")
#                 continue
        
#         # Update metadata for this ticker after processing all ranges
#         try:
#             print(f"  Updating metadata for {ticker}...")
#             update_metadata_for_symbol(ticker, interval)
#         except Exception as e:
#             print(f"  Failed to update metadata for {ticker}: {e}")

#     try:
#         conn.commit()
#         print("Database changes committed successfully.")
#     except Exception as e:
#         print(f"Failed to commit database changes: {e}")
    
#     conn.close()
#     print("\n✅ All tickers processed successfully.")

# def retrieve_stock_data(ticker_list: Union[str, List[str]], start_date: date = datetime.now().date() - timedelta(days=365), end_date: date = datetime.now().date(), db_path: str = config['PATHS']['STOCK_DATABASE'], interval: str = '1d') -> pd.DataFrame:
#     """
#     Retrieve stock data from database, downloading missing data if necessary.
#     """
#     # Connect to database
#     conn = sqlite3.connect(db_path)

#     # Handle single ticker as list
#     if isinstance(ticker_list, str):
#         ticker_list = [ticker_list]

#     placeholders = ','.join(['?'] * len(ticker_list))

#     query = f"""
#         SELECT ticker, date, open, high, low, close, volume
#         FROM {config['DATABASE'][interval]}
#         WHERE ticker IN ({placeholders})
#     """

#     params = list(ticker_list)

#     # Add optional date filtering
#     if start_date:
#         query += " AND date >= ?"
#         params.append(start_date.strftime("%Y-%m-%d"))
#     if end_date:
#         query += " AND date <= ?"
#         params.append(end_date.strftime("%Y-%m-%d"))

#     query += " ORDER BY ticker, date ASC"

#     # Execute the query
#     try:
#         stock_data = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])
#     except Exception as e:
#         print(f"Failed to retrieve data: {e}")
#         stock_data = pd.DataFrame()
        
#     if stock_data.empty:
#         print("No data found in database. Downloading from Yahoo Finance...")
#         fetch_and_store_multiple_stocks(ticker_list, start_date, end_date, interval=interval)
#         stock_data = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])       
    
#     conn.close()
#     return stock_data

# def get_index_symbols(csv_path: str = r"Data\INDEXES\Nifty Total Market.csv", read: bool = True, data: Optional[pd.DataFrame] = None, symbol_column: str = 'Symbol') -> List[str]:
#     """
#     Extract stock symbols from CSV file or DataFrame.
#     """
#     if read:
#         data = pd.read_csv(csv_path)
#     if data is None or symbol_column not in data.columns:
#         raise ValueError("CSV must contain a 'Symbol' column.")
#     symbols = [f"{sym.strip()}.NS" for sym in data[symbol_column]]
#     return symbols

# def fetch_data_for_symbols_df(symbols_df: pd.DataFrame, symbol_column: str = 'Symbol', start_date: Optional[date] = None, end_date: Optional[date] = None, interval: str = '1d'):
#     """
#     Fetch and store stock data for all symbols in a DataFrame using fetch_and_store_multiple_stocks.
#     """
#     print(f"Starting fetch_data_for_symbols_df...")
#     print(f"Input DataFrame shape: {symbols_df.shape}")
#     print(f"Symbol column: {symbol_column}")
    
#     try:
#         ticker_list = get_index_symbols(data=symbols_df, read=False, symbol_column=symbol_column)
#         print(f"Generated ticker list with {len(ticker_list)} symbols")
#         print(f"First 5 tickers: {ticker_list[:5]}")
        
#         if len(ticker_list) > 100:
#             print(f"Warning: Large number of tickers ({len(ticker_list)}). This may take a while.")
#             proceed = input("Do you want to continue? (y/n): ")
#             if proceed.lower() != 'y':
#                 print("Operation cancelled by user.")
#                 return
        
#         fetch_and_store_multiple_stocks(ticker_list, start_date=start_date, end_date=end_date, interval=interval)
        
#     except Exception as e:
#         print(f"Error in fetch_data_for_symbols_df: {e}")
#         import traceback
#         traceback.print_exc()


import sqlite3
from Config.config import config
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd


def fetch_and_store_multiple_stocks(ticker_list, start_date=None, end_date=None, db_path = config['PATHS']['STOCK_DATABASE'], interval='1d'):
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {config['DATABASE'][interval]} (
            ticker TEXT,
            date DATE,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.commit()

    # Download all tickers at once
    try:
        if not (start_date or end_date):
            stock_data = yf.download(ticker_list, period='max', group_by='ticker', auto_adjust=False, threads=True, interval=interval, end=datetime.now() - timedelta(days=0))
        else: 
            stock_data = yf.download(ticker_list, start=start_date, end=end_date, group_by='ticker', auto_adjust=False, threads=True, interval=interval)
    except Exception as e:
        print(f"Failed to download stock data: {e}")
        conn.close()
        return

    if stock_data is None or stock_data.empty:
        print("No data downloaded.")
        conn.close()
        return

    # Handle both single and multiple tickers correctly
    if len(ticker_list) == 1:
        # Single ticker case
        ticker = ticker_list[0]
        rows_to_insert = []
        # stock_data = stock_data.dropna(subset=['Open', 'High', 'Low', 'Close'])  # Drop rows with NaN values
        for index, row in stock_data.iterrows():
            rows_to_insert.append((
                ticker,
                index.strftime("%Y-%m-%d"),
                row['Open'],
                row['High'],
                row['Low'],
                row['Close'],
                row['Volume']
            ))
        cursor.executemany(f"""
            INSERT OR IGNORE INTO {config['DATABASE'][interval]} (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, rows_to_insert)
        print(f"Inserted {cursor.rowcount} rows for {ticker} into the database.")
    else:
        # Multiple tickers case
        for ticker in ticker_list:
            if ticker not in stock_data.columns.get_level_values(0).unique():
                print(f"No data for {ticker}. Skipping.")
                continue
            rows_to_insert = []
            ticker_data = stock_data[ticker]
            ticker_data = ticker_data.dropna(subset=['Open', 'High', 'Low', 'Close'])
            # ticker_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            for index, row in ticker_data.iterrows():
                rows_to_insert.append((
                    ticker,
                    index.strftime("%Y-%m-%d"),
                    row['Open'],
                    row['High'],
                    row['Low'],
                    row['Close'],
                    row['Volume']
                ))
            cursor.executemany(f"""
                INSERT OR IGNORE INTO {config['DATABASE'][interval]} (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows_to_insert)
            print(f"Inserted {cursor.rowcount} rows for {ticker} into the database.")

    conn.commit()
    conn.close()
    print("✅ All tickers processed successfully.")

def retrieve_stock_data(ticker_list, start_date=None, end_date=None, db_path = config['PATHS']['STOCK_DATABASE'], interval='1d'):
    # Connect to database
    conn = sqlite3.connect(db_path)

    # Handle single ticker as list
    if isinstance(ticker_list, str):
        ticker_list = [ticker_list]

    placeholders = ','.join(['?'] * len(ticker_list))  # Creates (?, ?, ?) dynamically based on number of tickers

    query = f"""
        SELECT ticker, date, open, high, low, close, volume
        FROM {config['DATABASE'][interval]}
        WHERE ticker IN ({placeholders})
    """

    params = list(ticker_list)

    # Add optional date filtering
    if start_date:
        query += " AND date >= ?"
        params.append(str(start_date))
    if end_date:
        query += " AND date <= ?"
        params.append(str(end_date))

    query += " ORDER BY ticker, date ASC"

    # Execute the query
    try:
        stock_data = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])
    except Exception as e:
        print(f"Failed to retrieve data: {e}")
        stock_data = pd.DataFrame()
        
    if stock_data.empty:
        print("No data found in database. Downloading from Yahoo Finance...")
        fetch_and_store_multiple_stocks(ticker_list, interval=interval)
        stock_data = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])       
    
    conn.close()

    return stock_data

def get_index_symbols(csv_path=r"Data\INDEXES\Nifty Total Market.csv", read=True, data=None, symbol_column='Symbol'):
    if read:
        data = pd.read_csv(csv_path)
    if data is None or symbol_column not in data.columns:
        raise ValueError("CSV must contain a 'Symbol' column.")
    symbols = [f"{sym.strip()}.NS" for sym in data[symbol_column]]
    return symbols

def fetch_data_for_symbols_df(symbols_df, symbol_column='Symbol', start_date=None, end_date=None, interval='1d'):
    """
    Fetch and store stock data for all symbols in a DataFrame using fetch_and_store_multiple_stocks.
    Args:
        symbols_df (pd.DataFrame): DataFrame containing stock symbols.
        symbol_column (str): Name of the column containing symbols.
        start_date (str): Start date for fetching data.
        end_date (str): End date for fetching data.
        interval (str): Data interval (e.g., '1d').
    """ 
    # Ticker List Example: ['INFY.NS', 'TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS', 'HINDUNILVR.NS']
    ticker_list = get_index_symbols(data=symbols_df, read=False, symbol_column=symbol_column)
    print(f"Ticker List: {ticker_list}")
    fetch_and_store_multiple_stocks(ticker_list, start_date=start_date, end_date=end_date, interval=interval)

import yfinance as yf
import json
import os
import sqlite3
from datetime import datetime
import pandas as pd
from Config.config import config
from typing import Union, List

# List of stock symbols to process
stock_list = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

# Path to output JSON file
metadata_file_path = "./Config/stock_metadata.json"
log_file_path = "./Config/failed_stocks.log"

def get_stock_metadata(stock_symbol, interval="1d"):
    """
    Download historical data for the stock and extract metadata.
    """
    try:
        query = f"""
            SELECT 
                MIN(date) as start_date,
                MAX(date) as end_date,
                COUNT(*) as total_records
            FROM {config['DATABASE'][interval]}
            WHERE ticker = ?
            """
        conn = sqlite3.connect(config['PATHS']['STOCK_DATABASE'])
        data = pd.read_sql_query(query, conn, params=[stock_symbol])    

        if data is None or data.empty:
            return None

        metadata = {
            "stock_symbol": stock_symbol,
            "start_date": data.iloc[0]['start_date'],
            "end_date": data.iloc[0]['end_date'],
            "total_records": data.iloc[0]['total_records'],
            "interval": interval,
            "last_updated": datetime.now().isoformat(timespec='seconds')
        }
        print('metadata', metadata)
        return metadata

    except Exception as e:
        print(f"Error fetching data for {stock_symbol}: {e}")
        log_failed_stock(stock_symbol)
        return None
        
    finally:
        conn.close()

def load_existing_metadata(file_path):
    """
    Load existing JSON metadata if it exists, or return empty dict.
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return {}
    except json.JSONDecodeError as e:
        print(f"JSON parsing error in metadata file: {e}")
        # Try to repair the file
        try:
            # Backup corrupted file
            import shutil
            backup_path = file_path + '.backup'
            shutil.copy2(file_path, backup_path)
            print(f"Backed up corrupted file to {backup_path}")
            
            # Create new empty metadata file
            with open(file_path, 'w') as f:
                f.write('{}')
            print(f"Created new empty metadata file.")
            return {}
        except Exception as backup_error:
            print(f"Failed to backup and recreate metadata file: {backup_error}")
            return {}
    except Exception as e:
        print(f"Error loading metadata file: {e}")
        return {}

def log_failed_stock(stock_symbol):
    """
    Log failed stock symbol to a separate file with timestamp.
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"{datetime.now().isoformat()} - Failed to fetch: {stock_symbol}\n")

def update_metadata_file(stock_list, file_path, interval="1d"):
    """
    Update the JSON metadata file with new or updated stock entries.
    """
    all_metadata = load_existing_metadata(file_path)

    for symbol in stock_list:
        metadata = get_stock_metadata(symbol, interval)
        if metadata:
            key = f"{symbol}_{interval}"
            all_metadata[key] = metadata

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    print(f"Metadata updated and saved to: {file_path}")


def get_database_metadata(ticker_list: Union[str, List[str]], db_path: str = config['PATHS']['STOCK_DATABASE'], interval: str = '1d') -> dict:
    """
    Extract metadata from database using SQL queries.
    Returns metadata with start_date, end_date, and total_records for each ticker.
    """
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Handle single ticker as list
    if isinstance(ticker_list, str):
        ticker_list = [ticker_list]
    
    metadata = {}
    
    for ticker in ticker_list:
        # Query to get metadata for this ticker
        query = f"""
            SELECT 
                MIN(date) as start_date,
                MAX(date) as end_date,
                COUNT(*) as total_records
            FROM {config['DATABASE'][interval]}
            WHERE ticker = ?
        """
        
        try:
            cursor.execute(query, (ticker,))
            result = cursor.fetchone()
            
            if result and result[0] is not None:
                start_date, end_date, total_records = result
                metadata[ticker] = {
                    "stock_symbol": ticker,
                    "start_date": start_date,
                    "end_date": end_date,
                    "total_records": total_records,
                    "interval": interval,
                    "last_updated": datetime.now().isoformat(timespec='seconds'),
                }
                
        except Exception as e:
            print(f"Error querying metadata for {ticker}: {e}")
    
    conn.close()
    return metadata

def get_all_database_metadata(db_path: str = config['PATHS']['STOCK_DATABASE'], interval: str = '1d') -> dict:
    """
    Get metadata for all tickers in the database.
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all unique tickers
    query = f"SELECT DISTINCT ticker FROM {config['DATABASE'][interval]} ORDER BY ticker"
    
    try:
        cursor.execute(query)
        tickers = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if tickers:
            return get_database_metadata(tickers, db_path, interval)
        else:
            return {}
            
    except Exception as e:
        print(f"Error getting ticker list: {e}")
        conn.close()
        return {}
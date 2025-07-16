import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os

def get_market_overview():
    """
    Fetch market overview data including major indices and their performance
    """
    try:
        indices = {
            "S&P 500": "^GSPC",
            "NASDAQ": "^IXIC", 
            "DOW": "^DJI",
            "Russell 2000": "^RUT",
            "VIX": "^VIX"
        }
        
        market_data = {}
        
        for name, symbol in indices.items():
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                previous = hist['Close'].iloc[-2] if len(hist) > 1 else current
                change = ((current - previous) / previous) * 100
                
                market_data[name] = {
                    'current': current,
                    'change': change,
                    'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
                }
        
        return market_data
    
    except Exception as e:
        print(f"Error fetching market overview: {e}")
        return {}

def get_portfolio_summary():
    """
    Generate portfolio summary data (mock implementation for template)
    """
    try:
        # In a real implementation, this would connect to portfolio management system
        portfolio_data = {
            'total_value': 125500000,  # $125.5M
            'daily_pnl': 287000,      # $287K
            'daily_pnl_pct': 0.023,   # 2.3%
            'sharpe_ratio': 2.14,
            'max_drawdown': -0.032,    # -3.2%
            'positions': 47,
            'cash_position': 0.05,     # 5%
            'sector_allocation': {
                'Technology': 0.285,
                'Healthcare': 0.162,
                'Financial': 0.148,
                'Consumer Discretionary': 0.123,
                'Industrials': 0.091,
                'Communication': 0.084,
                'Consumer Staples': 0.052,
                'Energy': 0.028,
                'Materials': 0.019,
                'Utilities': 0.006,
                'Real Estate': 0.002
            }
        }
        
        return portfolio_data
        
    except Exception as e:
        print(f"Error fetching portfolio summary: {e}")
        return {}

def fetch_stock_data(symbol, period="1y"):
    """
    Fetch comprehensive stock data for analysis
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Get historical data
        hist = ticker.history(period=period)
        
        # Get info
        info = ticker.info
        
        # Calculate technical indicators
        if not hist.empty:
            # Simple moving averages
            hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
            
            # RSI calculation
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            hist['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            hist['BB_Middle'] = hist['Close'].rolling(window=20).mean()
            bb_std = hist['Close'].rolling(window=20).std()
            hist['BB_Upper'] = hist['BB_Middle'] + (bb_std * 2)
            hist['BB_Lower'] = hist['BB_Middle'] - (bb_std * 2)
            
            # Volume indicators
            hist['Volume_SMA'] = hist['Volume'].rolling(window=20).mean()
            hist['Volume_Ratio'] = hist['Volume'] / hist['Volume_SMA']
        
        return {
            'history': hist,
            'info': info,
            'current_price': hist['Close'].iloc[-1] if not hist.empty else None,
            'symbol': symbol
        }
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def get_economic_data():
    """
    Fetch economic indicator data (requires API integration in production)
    """
    try:
        # This would typically connect to FRED, Bloomberg, or other economic data APIs
        # For template purposes, returning mock data structure
        economic_indicators = {
            'gdp_growth': {
                'current': 3.2,
                'previous': 2.8,
                'forecast': 3.0,
                'date': datetime.now() - timedelta(days=30)
            },
            'unemployment': {
                'current': 3.8,
                'previous': 3.9,
                'forecast': 3.7,
                'date': datetime.now() - timedelta(days=7)
            },
            'inflation_cpi': {
                'current': 2.8,
                'previous': 3.1,
                'forecast': 2.9,
                'date': datetime.now() - timedelta(days=14)
            },
            'fed_funds_rate': {
                'current': 5.25,
                'previous': 5.25,
                'forecast': 5.25,
                'date': datetime.now() - timedelta(days=45)
            }
        }
        
        return economic_indicators
        
    except Exception as e:
        print(f"Error fetching economic data: {e}")
        return {}

def get_options_data(symbol, expiry_date=None):
    """
    Fetch options data for volatility analysis
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Get available expiry dates
        expiry_dates = ticker.options
        
        if not expiry_dates:
            return None
        
        # Use provided expiry or nearest one
        if expiry_date is None:
            expiry_date = expiry_dates[0]
        elif expiry_date not in expiry_dates:
            expiry_date = expiry_dates[0]
        
        # Get options chain
        options_chain = ticker.option_chain(expiry_date)
        
        calls = options_chain.calls
        puts = options_chain.puts
        
        return {
            'calls': calls,
            'puts': puts,
            'expiry_date': expiry_date,
            'available_expiries': expiry_dates
        }
        
    except Exception as e:
        print(f"Error fetching options data for {symbol}: {e}")
        return None

def get_sector_performance():
    """
    Fetch sector ETF performance data
    """
    try:
        sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financial': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Industrials': 'XLI',
            'Communication Services': 'XLC',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Materials': 'XLB',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE'
        }
        
        sector_data = {}
        
        for sector, etf in sector_etfs.items():
            ticker = yf.Ticker(etf)
            hist = ticker.history(period="1y")
            
            if not hist.empty:
                # Calculate returns for different periods
                current_price = hist['Close'].iloc[-1]
                
                returns = {}
                periods = {'1D': 1, '1W': 5, '1M': 22, '3M': 66, 'YTD': len(hist)}
                
                for period_name, days in periods.items():
                    if len(hist) >= days:
                        past_price = hist['Close'].iloc[-days]
                        returns[period_name] = ((current_price - past_price) / past_price) * 100
                
                # Calculate volatility and momentum
                daily_returns = hist['Close'].pct_change().dropna()
                volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized
                
                # Volume analysis
                avg_volume = hist['Volume'].mean()
                recent_volume = hist['Volume'].iloc[-5:].mean()
                volume_ratio = recent_volume / avg_volume
                
                sector_data[sector] = {
                    'returns': returns,
                    'volatility': volatility,
                    'volume_ratio': volume_ratio,
                    'current_price': current_price,
                    'etf_symbol': etf
                }
        
        return sector_data
        
    except Exception as e:
        print(f"Error fetching sector performance: {e}")
        return {}

def get_alternative_data_sources():
    """
    Placeholder for alternative data sources integration
    In production, this would integrate with providers like:
    - Social sentiment APIs
    - Satellite data providers
    - News sentiment analysis
    - Economic nowcasting APIs
    """
    try:
        # Mock alternative data structure
        alt_data = {
            'social_sentiment': {
                'overall_market': 0.68,
                'trending_stocks': ['AAPL', 'TSLA', 'NVDA'],
                'sentiment_scores': {
                    'AAPL': 0.75,
                    'TSLA': 0.45,
                    'NVDA': 0.82
                }
            },
            'news_sentiment': {
                'positive_stories': 156,
                'negative_stories': 89,
                'neutral_stories': 203,
                'overall_sentiment': 0.52
            },
            'economic_nowcast': {
                'gdp_nowcast': 3.2,
                'recession_probability': 0.15,
                'inflation_nowcast': 2.8
            }
        }
        
        return alt_data
        
    except Exception as e:
        print(f"Error fetching alternative data: {e}")
        return {}

def validate_api_connection():
    """
    Validate API connections and data availability
    """
    try:
        # Test basic yfinance connection
        test_ticker = yf.Ticker("AAPL")
        test_data = test_ticker.history(period="1d")
        
        if test_data.empty:
            return False, "Unable to fetch test data from yfinance"
        
        return True, "API connections validated successfully"
        
    except Exception as e:
        return False, f"API validation failed: {e}"

import numpy as np
from datetime import datetime
import streamlit as st
import pytz
import pandas as pd

def display_market_status():
    import pandas_market_calendars as mcal
    nse = mcal.get_calendar('NSE')
    now = pd.Timestamp.now(tz='Asia/Kolkata')
    schedule = nse.schedule(start_date=now.date(), end_date=now.date())
    market_open = False
    if not schedule.empty:
        # Convert UTC to IST
        ist = pytz.timezone('Asia/Kolkata')
        market_open_time = schedule.iloc[0]['market_open'].tz_convert(ist)
        market_close_time = schedule.iloc[0]['market_close'].tz_convert(ist)
        market_open = (now >= market_open_time) and (now <= market_close_time)
    return market_open


def calculate_position_size(current_price, capital=100000, risk_per_trade=0.01):
    """
    Calculate the number of shares to buy and total investment amount
    based on current price, total capital, and risk per trade.

    Parameters:
    - current_price_series (pd.Series): Series containing the current price (e.g., last closing price)
    - capital (float): Total capital available for investment
    - risk_per_trade (float): Fraction of capital to risk on a single trade (e.g., 0.02 for 2%)

    Returns:
    - quantity (int): Number of shares to buy
    - investment_amount (float): Total investment amount
    """

    quantity = int((risk_per_trade * capital) / current_price)
    investment_amount = quantity * current_price

    return quantity, investment_amount



# def fractional_kelly(current_price, mu, sigma, r_f, b, capital=100000):
#     """
#     Calculate the optimal number of shares and investment amount using Fractional Kelly Criterion.

#     Parameters:
#     - current_price (float or array): Current price(s) of the stock(s)
#     - mu (float or array): Expected return(s)
#     - sigma (float or array): Volatility(ies)
#     - r_f (float): Risk-free rate
#     - b (float or array): Probability of winning (or edge)
#     - capital (float): Total capital

#     Returns:
#     - quantity (array): Number of shares to buy
#     - investment_amount (array): Amount to invest in each stock
#     """
#     mu = np.array(mu, dtype=float)
#     sigma = np.array(sigma, dtype=float)
#     b = np.array(b, dtype=float)
#     current_price = np.array(current_price, dtype=float)

#     # Basic Kelly formula adjusted conservatively
#     f_star = b * (mu - r_f) / (sigma**2)
#     f_fractional = f_star / (1 + f_star)
#     f_fractional = np.clip(f_fractional, 0, 1)  # avoid negative or overleveraged bets

#     # Investment and quantity
#     investment_amount = f_fractional * capital
#     quantity = np.floor(investment_amount / current_price).astype(int)

#     return quantity, investment_amount


def fractional_kelly(current_price, mu, sigma, r_f, b, capital=100000):
    """
    Calculate the optimal fraction of bankroll to allocate to each stock
    using the Fractional Kelly Criterion.

    Parameters:
    mu (array): expected returns of each stock
    sigma (array): volatilities of each stock
    r_f (float): risk-free rate
    b (array): odds or probabilities of each stock going up

    Returns:
    f (array): optimal fractions of bankroll to allocate to each stock
    """
    f = (b * (mu - r_f) / (sigma**2)) / (1 + (b * (mu - r_f) / (sigma**2)))
    return (f * capital) // current_price, f * capital

def calculate_daily_returns(df, oc_returns=True, col_name='daily_returns'):
    """
    Calculate daily returns for a DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame with 'open' and 'close' columns
    - oc_returns (bool): If True, use open-close returns; else, use close-close returns

    Returns:
    - pd.Series: Daily returns in percentage
    """
    if oc_returns:
        df[col_name] = ((df['close'] - df['open']) / df['open']) * 100
    else:
        df[col_name] = df['close'].pct_change() * 100
    return df

def get_current_week_str(interval='1d'):
    """
    Returns the current ISO week string in the format 'YYYY-Www',
    e.g., '2025-W25'.
    """
    if interval == '1d':
        now = datetime.now()
        year, week_num, _ = now.isocalendar()
        return f"{year}-W{week_num:02d}"
    elif interval == '1wk':
        # For weekly data, return the current year and week number
        now = datetime.now()
        return f"{now.year}-{now.month:02d}"
    else:
        raise ValueError("Unsupported interval. Use '1d' or '1wk'.")
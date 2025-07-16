import os
from ast import Pass
from turtle import position
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import itertools

# ===== CONFIGURATION =====
LOOKBACK_YEARS = 1               # Data period
EMA_SLOW = 55                    # Primary trend filter
EMA_FAST = 21                    # Phase detection
PEAK_TROUGH_WINDOW = 11          # For swing point detection
initial_capital = 10000000

# ===== DATA FETCHING =====
def fetch_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=LOOKBACK_YEARS*365)

    data = yf.download(PRIMARY_INDEX, start=start_date, end=end_date, group_by='Ticker')
    # print(data)
    if data is not None:
        return data.dropna()
    else:
        print("No data fetched.")
        return pd.DataFrame()


# ===== TREND ANALYSIS =====
def calculate_emas(df):
    df[('Price', f'{PRIMARY_INDEX}_{EMA_SLOW}_EMA')] = df[(PRIMARY_INDEX, 'Close')].ewm(span=EMA_SLOW).mean()
    df[('Price', f'{PRIMARY_INDEX}_{EMA_FAST}_EMA')] = df[(PRIMARY_INDEX, 'Close')].ewm(span=EMA_FAST).mean()
    return df

def find_peaks_troughs(series, window=PEAK_TROUGH_WINDOW):
    peaks = (series == series.rolling(window, center=True).max()) & (series != series.shift(1))
    troughs = (series == series.rolling(window, center=True).min()) & (series != series.shift(1))
    return peaks, troughs


def identify_phases(df):
    """Classify market phases as Bull or Bear only"""
    df['Peak'], df['Trough'] = find_peaks_troughs(df[(PRIMARY_INDEX, 'Close')])
    # print('Peak Trough', df)

    phases = []
    for i in range(len(df)):
        # Primary Trend with EMA crossover condition
        close = df[(PRIMARY_INDEX, 'Close')].iloc[i].item()
        ema_50 = df[('Price', f'{PRIMARY_INDEX}_{EMA_SLOW}_EMA')].iloc[i].item()
        ema_20 = df[('Price', f'{PRIMARY_INDEX}_{EMA_FAST}_EMA')].iloc[i].item()

        # For Bull: price > 50EMA and 20EMA > 50EMA (crossed above)
        # For Bear: price <= 50EMA and 20EMA < 50EMA (crossed below)
        if (close > ema_50) and (ema_20 > ema_50):
            phase = "Bull"
        elif (close <= ema_50) and (ema_20 < ema_50):
            phase = "Bear"
        else:
            # If neither, keep previous phase if available, else default to Bear
            phase = phases[-1] if phases else "Bear"
        phases.append(phase)

    df['Phase'] = phases
    return df

# ===== SIGNAL GENERATION =====
def generate_signals(df):
    # Add trend structure columns before signal logic
    df['Higher_High'] = df['Peak'] & (df[(PRIMARY_INDEX, 'Close')] > df[(PRIMARY_INDEX, 'Close')].shift(1).rolling(5).max())
    df['Higher_Low'] = df['Trough'] & (df[(PRIMARY_INDEX, 'Close')] > df[(PRIMARY_INDEX, 'Close')].shift(1).rolling(5).min())
    df['Lower_High'] = df['Peak'] & (df[(PRIMARY_INDEX, 'Close')] < df[(PRIMARY_INDEX, 'Close')].shift(1).rolling(5).max())
    df['Lower_Low'] = df['Trough'] & (df[(PRIMARY_INDEX, 'Close')] < df[(PRIMARY_INDEX, 'Close')].shift(1).rolling(5).min())

    signals = []
    hh = []  # Higher Highs tracker
    hl = []  # Higher Lows tracker
    lh = []  # Lower Highs tracker
    ll = []  # Lower Lows tracker

    for i in range(1, len(df)):
        price = df[(PRIMARY_INDEX, 'Close')].iloc[i].item()
        phase = df['Phase'].iloc[i]
        # Track structure
        if df['Higher_High'].iloc[i]:
            hh.append(price)
        if df['Higher_Low'].iloc[i]:
            hl.append(price)
        if df['Lower_High'].iloc[i]:
            lh.append(price)
        if df['Lower_Low'].iloc[i]:
            ll.append(price)

        # Signal logic
        if phase == "Bull":
            # Buy if price > recent HH, Sell if price < recent HL
            if hh and price > hh[-1]:
                signals.append("BUY")
            # elif hl and price < hl[-1]:
            #     # signals.append("SELL")
            #     pass
            else:
                signals.append("HOLD")
        # elif phase == "Bear":
        #     # Sell if price < recent LL, Buy if price > recent LH
        #     if ll and price < ll[-1]:
        #         signals.append("SELL")
        #     elif lh and price > lh[-1]:
        #         signals.append("BUY")
        #         # pass
        #     else:
        #         signals.append("HOLD")
        else:
            signals.append("HOLD")


    # Ensure signals match length
    signals.insert(0, "HOLD")
    df['Signal'] = signals[:len(df)]
    # Print the counts of each signal type
    print('Signal counts:', df['Signal'].value_counts())
    return df

# ===== VISUALIZATION =====
def plot_dow_theory(df):
    plt.figure(figsize=(16,12))
    ax = plt.subplot(2,1,1)

    # Price and EMAs
    plt.plot(df[(PRIMARY_INDEX, 'Close')], label=PRIMARY_INDEX, color='#1f77b4')
    plt.plot(df[('Price', f'{PRIMARY_INDEX}_{EMA_SLOW}_EMA')], '--', label= 'SLOW EMA', color='black')
    plt.plot(df[('Price', f'{PRIMARY_INDEX}_{EMA_FAST}_EMA')], ':', label='FAST EMA', color='purple')

    # Mark Peaks/Troughs
    peaks = df[df['Peak']]
    troughs = df[df['Trough']]
    plt.scatter(peaks.index, peaks[(PRIMARY_INDEX, 'Close')], color='red', marker='v', s=100, label='Peak')
    plt.scatter(troughs.index, troughs[(PRIMARY_INDEX, 'Close')], color='green', marker='^', s=100, label='Trough')

    # Signals
    buys = df[df['Signal'] == "BUY"]
    sells = df[df['Signal'] == "SELL"]
    plt.scatter(buys.index, buys[(PRIMARY_INDEX, 'Close')], color='green', marker='*', s=60, label='Buy Signal')
    plt.scatter(sells.index, sells[(PRIMARY_INDEX, 'Close')], color='red', marker='*', s=60, label='Sell Signal')

    # --- PHASE VISUALIZATION ---
    phase_colors = {
        'Bear': '#e53935',  # Red
        'Bull': '#43a047'   # Green
    }
    last_phase = None
    phase_start = None
    for i, (idx, row) in enumerate(df.iterrows()):
        phase = row['Phase']
        if isinstance(phase, pd.Series):
            phase = phase.iloc[0]
        if isinstance(last_phase, pd.Series):
            last_phase = last_phase.iloc[0]
        if phase != last_phase:
            if last_phase is not None and phase_start is not None:
                plt.axvspan(phase_start, idx, color=phase_colors.get(last_phase, '#eeeeee'), alpha=0.25, label=last_phase if last_phase not in plt.gca().get_legend_handles_labels()[1] else "")
            phase_start = idx
            last_phase = phase
    # Add the last phase
    if last_phase is not None and phase_start is not None:
        plt.axvspan(phase_start, df.index[-1], color=phase_colors.get(last_phase, '#eeeeee'), alpha=0.25, label=last_phase if last_phase not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.title("Dow Theory Analysis: Nifty 50 with Market Phases")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Trend Structure Plot
    ax2 = plt.subplot(2,1,2, sharex=ax)
    plt.plot(df['Higher_High'].astype(int), label='Higher High', color='green')
    plt.plot(df['Higher_Low'].astype(int), label='Higher Low', color='lime')
    plt.plot(df['Lower_High'].astype(int), label='Lower High', color='orange')
    plt.plot(df['Lower_Low'].astype(int), label='Lower Low', color='red')
    plt.title("Trend Structure")
    plt.yticks([0,1], ["False", "True"])
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Backtesting Framework
def backtest_swing_strategy(df_dict, initial_capital=initial_capital, leverage=4, max_allocation_pct=1, debug=True,
                            stoploss_pct=0.02, target_pct=0.05, max_holding_days=5):
    """
    Backtest strategy for multiple stocks simultaneously
    
    Args:
        df_dict: Dictionary of DataFrames, key=stock_symbol, value=DataFrame
        initial_capital: Starting capital
        leverage: Leverage multiplier
        max_allocation_pct: Maximum allocation per stock as % of capital
        debug: Whether to show step-by-step visualization
        stoploss_pct: Stop loss percentage
        target_pct: Target profit percentage
        max_holding_days: Maximum holding period
    """
    capital = initial_capital
    positions = {}  # Dictionary: stock_symbol -> list of positions
    trade_log = []
    
    # Initialize positions dictionary for each stock
    for stock_symbol in df_dict.keys():
        positions[stock_symbol] = []

    equity_curve = []
    dates = []
    
    # Get the common date range across all stocks
    all_dates = set()
    for stock_symbol, df in df_dict.items():
        all_dates.update(df['Date'].tolist())
    common_dates = sorted(list(all_dates))
    
    print(f"Backtesting {len(df_dict)} stocks from {common_dates[0]} to {common_dates[-1]}")
    print(f"Stocks: {list(df_dict.keys())}")

    for i, current_date in enumerate(common_dates):
        daily_capital = capital
        
        # Process each stock for the current date
        for stock_symbol, df in df_dict.items():
            # Find the row for current date in this stock's DataFrame
            stock_data = df[df['Date'] == current_date]
            if stock_data.empty:
                continue
                
            row = stock_data.iloc[0]
            
            # Extract prices and signal for this stock
            open_price = float(row[stock_symbol]['Open'])
            high = float(row[stock_symbol]['High'])
            low = float(row[stock_symbol]['Low'])
            close = float(row[stock_symbol]['Close'])
            signal = str(row['Signal']['']).strip().upper()

            # Check existing positions for this stock
            for pos in positions[stock_symbol][:]:  # use a copy to modify safely
                pos['days_held'] += 1
                hit_sl = low <= pos['sl']
                hit_target = high >= pos['target']
                max_days = pos['days_held'] >= max_holding_days

                if hit_sl or hit_target or signal == 'SELL' or max_days:
                    # Determine exit price
                    if hit_sl:
                        exit_price = pos['sl']
                    elif hit_target:
                        exit_price = pos['target']
                    else:
                        exit_price = close
                
                    charges = ((((pos['total_investment'] - pos['investment']) * 0.15) / 365) * pos['days_held']) + 30
                    pnl = (exit_price - pos['entry']) * pos['qty'] 
                    capital += pos['investment'] + pnl - charges
                    daily_capital += pos['investment'] + pnl - charges
                    
                    trade_log.append({
                        'Stock': stock_symbol,
                        'Entry Date': pos['entry_date'],
                        'Exit Date': current_date,
                        'Entry': pos['entry'],
                        'Exit': exit_price,
                        'Qty': pos['qty'],
                        'Our Investment': pos['investment'],
                        'Margin Investment': pos['total_investment'],
                        'Holding Days': pos['days_held'],
                        'P&L': pnl,
                        'P&L Pct': pnl / pos['total_investment'],
                        'Capital After Trade Close': capital,
                        'Signal': signal,
                        'Reason': 'SL' if hit_sl else 'Target' if hit_target else 'Max Days' if max_days else signal,
                        'Charges': charges
                    })

                    # Print details for closed trades
                    print("-"*50)
                    print(f"TRADE CLOSED - {stock_symbol} on {current_date}")
                    print(f"Entry Date: {pos['entry_date']}")
                    print(f"Entry Price: {pos['entry']:.2f}")
                    print(f"Exit Price: {exit_price:.2f}")
                    print(f"Quantity: {pos['qty']}")
                    print(f"Holding Days: {pos['days_held']}")
                    print(f"P&L: {pnl:.2f} ({(pnl/pos['total_investment'])*100:.2f}%)")
                    print(f"Charges: {charges:.2f}")
                    print(f"Capital After Trade Close: {capital:.2f}")
                    print(f"Reason: {'SL' if hit_sl else 'Target' if hit_target else 'Max Days' if max_days else signal}")
                    print("-"*50)

                    positions[stock_symbol].remove(pos)

            # Process buy signal for this stock
            if signal == 'BUY':
                if capital > open_price:
                    alloc = capital * max_allocation_pct
                    total_invest = alloc * leverage
                    qty = total_invest // open_price
                else:
                    print('No Stock to Buy as we have low capital')
                    qty = 0

                if qty > 0:
                    positions[stock_symbol].append({
                        'entry_date': current_date,
                        'entry': open_price,
                        'sl': open_price * (1 - stoploss_pct),
                        'target': open_price * (1 + target_pct),
                        'qty': qty,
                        'investment': alloc,
                        'total_investment': total_invest,
                        'days_held': 0
                    })
                    capital -= alloc
                    daily_capital -= alloc
                    
                    # Print trade details
                    print("="*50)
                    print(f"BUY SIGNAL - {stock_symbol} on {current_date}")
                    print(f"Entry Price: {open_price:.2f}")
                    print(f"Quantity: {qty}")
                    print(f"Stoploss: {open_price * (1 - stoploss_pct):.2f}")
                    print(f"Target: {open_price * (1 + target_pct):.2f}")
                    print(f"Investment: {alloc:.2f}")
                    print(f"Total (with leverage): {total_invest:.2f}")
                    print(f"Capital after buy: {capital:.2f}")
                    print("="*50)

        # Update equity curve with daily capital
        equity_curve.append(daily_capital)
        dates.append(current_date)

        # Debug visualization
        if debug and i % 5 == 0:  # Show every 5th day for performance
            plt.clf()
            
            # Plot 1: Portfolio Overview
            plt.subplot(2,2,1)
            plt.plot(dates, equity_curve, label='Portfolio Value', color='navy', linewidth=2)
            plt.title(f"Portfolio Performance - Day {i+1}")
            plt.xlabel("Date")
            plt.ylabel("Capital")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Individual Stock Prices
            plt.subplot(2,2,2)
            for stock_symbol, df in df_dict.items():
                stock_dates = df['Date'].tolist()
                stock_prices = df[stock_symbol]['Close'].tolist()
                # Only plot up to current date
                valid_indices = [j for j, d in enumerate(stock_dates) if d <= current_date]
                if valid_indices:
                    plot_dates = [stock_dates[j] for j in valid_indices]
                    plot_prices = [stock_prices[j] for j in valid_indices]
                    plt.plot(plot_dates, plot_prices, label=stock_symbol, alpha=0.7)
            plt.title("Stock Prices")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Open Positions Summary
            plt.subplot(2,2,3)
            position_summary = []
            for stock_symbol, pos_list in positions.items():
                for pos in pos_list:
                    position_summary.append({
                        'Stock': stock_symbol,
                        'Entry': pos['entry'],
                        'Current': pos['entry'],  # Simplified for now
                        'P&L': 0  # Would need current price calculation
                    })
            
            if position_summary:
                stocks = [p['Stock'] for p in position_summary]
                entries = [p['Entry'] for p in position_summary]
                plt.bar(stocks, entries, alpha=0.7)
                plt.title("Open Positions")
                plt.ylabel("Entry Price")
                plt.xticks(rotation=45)
            
            # Plot 4: Daily Summary
            plt.subplot(2,2,4)
            active_positions = sum(len(pos_list) for pos_list in positions.values())
            plt.text(0.1, 0.8, f"Date: {current_date}", fontsize=12)
            plt.text(0.1, 0.6, f"Capital: ₹{daily_capital:,.2f}", fontsize=12)
            plt.text(0.1, 0.4, f"Active Positions: {active_positions}", fontsize=12)
            plt.text(0.1, 0.2, f"Total Trades: {len(trade_log)}", fontsize=12)
            plt.axis('off')
            plt.title("Daily Summary")
            
            plt.tight_layout()
            plt.pause(0.1)

    # Final exit for all remaining positions
    for stock_symbol, pos_list in positions.items():
        for pos in pos_list:
            # Get the last available price for this stock
            last_row = df_dict[stock_symbol].iloc[-1]
            close_price = float(last_row[stock_symbol]['Close'])
            pnl = (close_price - pos['entry']) * pos['qty']
            charges = ((((pos['total_investment'] - pos['investment']) * 0.15) / 365) * pos['days_held']) + 30
            capital += pos['investment'] + pnl - charges
            
            trade_log.append({
                'Stock': stock_symbol,
                'Entry Date': pos['entry_date'],
                'Exit Date': last_row['Date'],
                'Entry': pos['entry'],
                'Exit': close_price,
                'Qty': pos['qty'],
                'Our Investment': pos['investment'],
                'Margin Investment': pos['total_investment'],
                'Holding Days': pos['days_held'],
                'P&L': pnl,
                'P&L Pct': pnl / pos['investment'],
                'Capital After Trade Close': capital,
                'Signal': 'FINAL_EXIT',
                'Reason': 'Final Exit',
                'Charges': charges
            })

            print("-"*50)
            print(f"FINAL EXIT - {stock_symbol}")
            print(f"Entry Date: {pos['entry_date']}")
            print(f"Entry Price: {pos['entry']:.2f}")
            print(f"Exit Price: {close_price:.2f}")
            print(f"Quantity: {pos['qty']}")
            print(f"Holding Days: {pos['days_held']}")
            print(f"P&L: {pnl:.2f} ({(pnl/pos['total_investment'])*100:.2f}%)")
            print(f"Charges: {charges:.2f}")
            print(f"Capital After Trade Close: {capital:.2f}")
            print("-"*50)

    results = pd.DataFrame(trade_log)
    if not results.empty and 'P&L' in results.columns:
        results['Cumulative P&L'] = results['P&L'].cumsum()
    else:
        results['Cumulative P&L'] = []
    
    return results, capital

def optimize_parameters(df, stoploss_range, target_range, holding_days_range):
    # df = df.reset_index()
    results_list = []
    for stoploss, target, holding in itertools.product(stoploss_range, target_range, holding_days_range):
        print(f'\nParameters:: Stoploss {stoploss}   Target {target}  Max Holding Period {holding}')
        res, final_cap = backtest_swing_strategy(
            df,
            initial_capital=initial_capital,
            stoploss_pct=stoploss,
            target_pct=target,
            max_holding_days=holding
        )
        print(f'RESULT: {res}')
        print(f'Final Capital: {final_cap}')
        # Calculate win rate and CAGR
        if not res.empty and 'P&L' in res.columns:
            num_wins = (res['P&L'] > 0).sum()
            num_trades = len(res)
            win_rate = num_wins / num_trades if num_trades > 0 else 0
        else:
            win_rate = 0
        start_date = pd.to_datetime(df['Date'].iloc[0])
        end_date = pd.to_datetime(df['Date'].iloc[-1])
        years = (end_date - start_date).days / 365.25
        if final_cap > 0 and years > 0:
            cagr = (final_cap / initial_capital) ** (1 / years) - 1
        results_list.append({
            'stoploss_pct': stoploss,
            'target_pct': target,
            'max_holding_days': holding,
            'final_capital': f'{final_cap:,.2f}',
            'win_rate': f'{win_rate:.2%}',
            'CAGR': f'{cagr:.2%}'
        })
        print(f'CAGR: {cagr:.2%}     WIN Rate: {win_rate:.2%}')
    results_df = pd.DataFrame(results_list)
    # Choose the best by highest final_capital (or CAGR)
    best_row = results_df.loc[results_df['final_capital'].idxmax()]
    print('Optimization Results:')
    print(results_df.sort_values('final_capital', ascending=False).head(10))
    print('\nBest Parameters:', best_row)
    return best_row, results_df

def save_trade_log(results, primary_index):
    trade_log_path = f'trade_log_{primary_index}.csv'
    if os.path.exists(trade_log_path):
        os.remove(trade_log_path)
    results.to_csv(trade_log_path, index=False)

# ===== HELPER FUNCTIONS =====
def prepare_multiple_stock_data(stock_symbols):
    """
    Prepare data for multiple stocks
    
    Args:
        stock_symbols: List of stock symbols (e.g., ['RELIANCE.NS', 'TCS.NS', 'INFY.NS'])
    
    Returns:
        Dictionary of DataFrames, key=stock_symbol, value=processed_DataFrame
    """
    df_dict = {}
    
    for stock_symbol in stock_symbols:
        print(f"Processing {stock_symbol}...")
        
        # Set the global PRIMARY_INDEX for this stock
        global PRIMARY_INDEX
        PRIMARY_INDEX = stock_symbol
        
        # Fetch data for this stock
        df = fetch_data()
        
        if not df.empty:
            # Process the data (calculate EMAs, identify phases, generate signals)
            df = calculate_emas(df)
            df = identify_phases(df)
            df = generate_signals(df)
            df = df.reset_index()
            
            # Store in dictionary
            df_dict[stock_symbol] = df
            print(f"✓ {stock_symbol}: {len(df)} days of data processed")
        else:
            print(f"✗ {stock_symbol}: No data available")
    
    return df_dict

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("Running Enhanced Dow Theory Analyzer for Multiple Stocks...")

    while True:
        # Get multiple stock symbols from user
        stock_input = input('\nEnter NSE Stock Names (comma-separated, e.g., RELIANCE.NS,TCS.NS,INFY.NS): ')
        stock_symbols = [s.strip() for s in stock_input.split(',') if s.strip()]
        
        if not stock_symbols:
            print("No valid stock symbols provided.")
            continue
            
        print(f"Processing {len(stock_symbols)} stocks: {stock_symbols}")
        
        # Prepare data for all stocks
        df_dict = prepare_multiple_stock_data(stock_symbols)
        
        if not df_dict:
            print("Error: No data could be fetched for any of the specified stocks.")
            continue
            
        print(f"\nSuccessfully prepared data for {len(df_dict)} stocks")
        
        # Visualize individual stock analysis (optional)
        show_individual_plots = input("\nShow individual stock analysis plots? (y/n): ").lower().startswith('y')
        if show_individual_plots:
            for stock_symbol, df in df_dict.items():
                print(f"\nPlotting analysis for {stock_symbol}...")
                global PRIMARY_INDEX
                PRIMARY_INDEX = stock_symbol
                plot_dow_theory(df)
        
        # Run multi-stock backtest
        print(f"\nStarting multi-stock backtest...")
        results, final_capital = backtest_swing_strategy(df_dict)
        
        print(f"\n=== BACKTEST RESULTS ===")
        print(f"Final capital: ₹{final_capital:,.2f}")
        
        # Calculate and print portfolio statistics
        if not results.empty and 'P&L' in results.columns:
            # Overall statistics
            total_trades = len(results)
            winning_trades = (results['P&L'] > 0).sum()
            losing_trades = (results['P&L'] <= 0).sum()
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_pnl = results['P&L'].sum()
            avg_win = results[results['P&L'] > 0]['P&L'].mean() if winning_trades > 0 else 0
            avg_loss = results[results['P&L'] <= 0]['P&L'].mean() if losing_trades > 0 else 0
            
            print(f"Total Trades: {total_trades}")
            print(f"Winning Trades: {winning_trades}")
            print(f"Losing Trades: {losing_trades}")
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Total P&L: ₹{total_pnl:,.2f}")
            print(f"Average Win: ₹{avg_win:,.2f}")
            print(f"Average Loss: ₹{avg_loss:,.2f}")
            
            # Per-stock statistics
            print(f"\n=== PER-STOCK STATISTICS ===")
            for stock_symbol in df_dict.keys():
                stock_results = results[results['Stock'] == stock_symbol]
                if not stock_results.empty:
                    stock_trades = len(stock_results)
                    stock_wins = (stock_results['P&L'] > 0).sum()
                    stock_pnl = stock_results['P&L'].sum()
                    stock_win_rate = stock_wins / stock_trades if stock_trades > 0 else 0
                    print(f"{stock_symbol}: {stock_trades} trades, {stock_wins} wins, ₹{stock_pnl:,.2f} P&L, {stock_win_rate:.2%} win rate")
            
            # Calculate CAGR
            if len(df_dict) > 0:
                # Use the first stock's date range for CAGR calculation
                first_stock_df = list(df_dict.values())[0]
                start_date = pd.to_datetime(first_stock_df['Date'].iloc[0])
                end_date = pd.to_datetime(first_stock_df['Date'].iloc[-1])
                years = (end_date - start_date).days / 365.25
                if final_capital > 0 and years > 0:
                    cagr = (final_capital / initial_capital) ** (1 / years) - 1
                    print(f"\nCAGR: {cagr:.2%} per annum")
        else:
            print("No trades to analyze.")
        
        # Save detailed results
        save_results = input("\nSave detailed results to CSV? (y/n): ").lower().startswith('y')
        if save_results:
            results.to_csv('multi_stock_backtest_results.csv', index=False)
            print("Results saved to 'multi_stock_backtest_results.csv'")
        
        # Ask if user wants to continue
        continue_testing = input("\nTest another set of stocks? (y/n): ").lower().startswith('y')
        if not continue_testing:
            break
    
    print("Backtesting completed!")
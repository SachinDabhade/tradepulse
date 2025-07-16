import os
from ast import Pass
from turtle import position
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import itertools
import concurrent.futures
import json

# ===== CONFIGURATION =====
LOOKBACK_YEARS = 1               # Data period
EMA_SLOW = 55                    # Primary trend filter
EMA_FAST = 21                    # Phase detection
PEAK_TROUGH_WINDOW = 11          # For swing point detection
initial_capital = 100000

# ===== DATA FETCHING =====
def fetch_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=LOOKBACK_YEARS*365)

    data = yf.download(PRIMARY_INDEX, period='max', group_by='Ticker')
    # data = yf.download(PRIMARY_INDEX, end=end_date, group_by='Ticker')
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
def backtest_swing_strategy(df, initial_capital=initial_capital, leverage=1, max_allocation_pct=1,
                            stoploss_pct=0.02, target_pct=0.05, max_holding_days=5):
    # df = df.copy().reset_index(drop=True)
    capital = initial_capital
    positions = []
    trade_log = []

    equity_curve = []
    dates = []

    for i in range(len(df)):
        row = df.iloc[i]
        # Make sure we extract just the scalar datetime, not a Series
        date = row['Date']
        if isinstance(date, pd.Series):
            date = date.iloc[0]

        # ✅ Ensure prices are floats
        open_price = float(row[PRIMARY_INDEX]['Open'])
        high = float(row[PRIMARY_INDEX]['High'])
        low = float(row[PRIMARY_INDEX]['Low'])
        close = float(row[PRIMARY_INDEX]['Close'])
        signal = str(row['Signal']['']).strip().upper()

        # ✅ Check existing positions
        for pos in positions[:]:  # use a copy to modify safely
            pos['days_held'] += 1
            hit_sl = low <= pos['sl']
            hit_target = high >= pos['target']
            max_days = pos['days_held'] >= max_holding_days

            if hit_sl or hit_target or signal == 'SELL' or max_days:
                # ✅ Ensure exit_price is float
                if hit_sl:
                    exit_price = pos['sl']
                elif hit_target:
                    exit_price = pos['target']
                else:
                    exit_price = close
            
                charges = ((((pos['total_investment'] - pos['investment']) * 0.15) / 365) * pos['days_held']) + 30
                pnl = (exit_price - pos['entry']) * pos['qty'] 
                capital += pos['investment'] + pnl - charges
                trade_log.append({
                    'Entry Date': pos['entry_date'],
                    'Exit Date': date,
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

                # Print details for closed trades using the pos variable
                # print("-"*50)
                # print(f"TRADE CLOSED on {date}")
                # print(f"Entry Date: {pos['entry_date']}")
                # print(f"Entry Price: {pos['entry']:.2f}")
                # print(f"Exit Price: {exit_price:.2f}")
                # print(f"Quantity: {pos['qty']}")
                # print(f"Holding Days: {pos['days_held']}")
                # print(f"P&L: {(exit_price - pos['entry']) * pos['qty']:.2f} ({((exit_price - pos['entry']) * pos['qty'] / pos['total_investment'])*100:.2f}%)")
                # print(f"Charges: {charges:.2f}")
                # print(f"Capital After Trade Close: {capital:.2f}")
                # print(f"Reason: {'SL' if hit_sl else 'Target' if hit_target else 'Max Days' if max_days else signal}")
                # print("-"*50)

                positions.remove(pos)  # ✅ This now works since all are scalars
        
        # ✅ Buy Signal
        if signal == 'BUY':
            if capital > open_price:
                alloc = capital * max_allocation_pct
                total_invest = alloc * leverage
                qty = total_invest // open_price
            else:
                qty = 0

            if qty > 0:
                positions.append({
                    'entry_date': date,
                    'entry': open_price,
                    'sl': open_price * (1 - stoploss_pct),
                    'target': open_price * (1 + target_pct),
                    'qty': qty,
                    'investment': alloc,
                    'total_investment': total_invest,
                    'days_held': 0
                })
                capital -= alloc  # subtract actual investment
                # Print trade details in a pretty format when a trade is executed
                # print("="*50)
                # print(f"BUY SIGNAL on {date}")
                # print(f"Entry Price: {open_price:.2f}")
                # print(f"Quantity: {qty}")
                # print(f"Stoploss: {open_price * (1 - stoploss_pct):.2f}")
                # print(f"Target: {open_price * (1 + target_pct):.2f}")
                # print(f"Investment: {alloc:.2f}")
                # print(f"Total (with leverage): {total_invest:.2f}")
                # print(f"Capital after buy: {capital:.2f}")
                # print("="*50)

        equity_curve.append(capital)
        dates.append(date)

        # # Visualization (every day, or every N days for speed)
        # plt.clf()
        # plt.subplot(2,1,1)
        # plt.plot(df['Date'][:i+1], df[(PRIMARY_INDEX, 'Close')][:i+1], label='Close', color='black')

        # # Plot EMAs if available
        # if ('Price', f'{PRIMARY_INDEX}_{EMA_SLOW}_EMA') in df.columns:
        #     plt.plot(df['Date'][:i+1], df[('Price', f'{PRIMARY_INDEX}_{EMA_SLOW}_EMA')][:i+1], '--', label='SLOW EMA', color='gray', alpha=0.5)
        # if ('Price', f'{PRIMARY_INDEX}_{EMA_FAST}_EMA') in df.columns:
        #     plt.plot(df['Date'][:i+1], df[('Price', f'{PRIMARY_INDEX}_{EMA_FAST}_EMA')][:i+1], ':', label='FAST EMA', color='purple', alpha=0.5)

        # # Mark Peaks and Troughs
        # peaks = df.iloc[:i+1][df.iloc[:i+1]['Peak']]
        # troughs = df.iloc[:i+1][df.iloc[:i+1]['Trough']]
        # plt.scatter(peaks['Date'], peaks[(PRIMARY_INDEX, 'Close')], color='red', marker='v', s=80, label='Peak')
        # plt.scatter(troughs['Date'], troughs[(PRIMARY_INDEX, 'Close')], color='green', marker='^', s=80, label='Trough')

        # # Mark HH, HL, LH, LL
        # if 'Higher_High' in df.columns:
        #     hh = df.iloc[:i+1][df.iloc[:i+1]['Higher_High']]
        #     plt.scatter(hh['Date'], hh[(PRIMARY_INDEX, 'Close')], color='blue', marker='P', s=70, label='HH')
        # if 'Higher_Low' in df.columns:
        #     hl = df.iloc[:i+1][df.iloc[:i+1]['Higher_Low']]
        #     plt.scatter(hl['Date'], hl[(PRIMARY_INDEX, 'Close')], color='cyan', marker='X', s=70, label='HL')
        # if 'Lower_High' in df.columns:
        #     lh = df.iloc[:i+1][df.iloc[:i+1]['Lower_High']]
        #     plt.scatter(lh['Date'], lh[(PRIMARY_INDEX, 'Close')], color='orange', marker='P', s=70, label='LH')
        # if 'Lower_Low' in df.columns:
        #     ll = df.iloc[:i+1][df.iloc[:i+1]['Lower_Low']]
        #     plt.scatter(ll['Date'], ll[(PRIMARY_INDEX, 'Close')], color='magenta', marker='X', s=70, label='LL')

        # # Mark Buy/Sell signals
        # if 'Signal' in df.columns:
        #     buys = df.iloc[:i+1][df.iloc[:i+1]['Signal'] == "BUY"]
        #     sells = df.iloc[:i+1][df.iloc[:i+1]['Signal'] == "SELL"]
        #     plt.scatter(buys['Date'], buys[(PRIMARY_INDEX, 'Close')], color='lime', marker='*', s=120, label='Buy Signal')
        #     plt.scatter(sells['Date'], sells[(PRIMARY_INDEX, 'Close')], color='red', marker='*', s=120, label='Sell Signal')

        # # Plot open positions' entry, sl, target
        # # Plot open positions' entry, sl, target
        # for pos in positions:
        #     plt.axhline(pos['entry'], color='g', linestyle='--', alpha=0.5, label='Entry' if 'Entry' not in plt.gca().get_legend_handles_labels()[1] else "")
        #     plt.axhline(pos['sl'], color='r', linestyle=':', alpha=0.5, label='SL' if 'SL' not in plt.gca().get_legend_handles_labels()[1] else "")
        #     plt.axhline(pos['target'], color='b', linestyle=':', alpha=0.5, label='Target' if 'Target' not in plt.gca().get_legend_handles_labels()[1] else "")

        # # --- Plot peaks and troughs as they are detected (to visualize their "lateness") ---
        # if 'Peak' in df.columns and 'Trough' in df.columns:
        #     # Peaks/troughs as detected up to this day
        #     peaks = df.iloc[:i+1][df.iloc[:i+1]['Peak']]
        #     troughs = df.iloc[:i+1][df.iloc[:i+1]['Trough']]
        #     # Peaks/troughs that are "confirmed" (i.e., the current day is when the peak/trough is marked)
        #     if not peaks.empty:
        #         plt.scatter(peaks['Date'], peaks[(PRIMARY_INDEX, 'Close')], color='red', marker='v', s=100, label='Peak' if 'Peak' not in plt.gca().get_legend_handles_labels()[1] else "")
        #         # Draw a vertical line to show when the peak is detected (to visualize the lag)
        #         for _, peak_row in peaks.iterrows():
        #             plt.axvline(peak_row['Date'], color='red', linestyle='--', alpha=0.2)
        #     if not troughs.empty:
        #         plt.scatter(troughs['Date'], troughs[(PRIMARY_INDEX, 'Close')], color='green', marker='^', s=100, label='Trough' if 'Trough' not in plt.gca().get_legend_handles_labels()[1] else "")
        #         for _, trough_row in troughs.iterrows():
        #             plt.axvline(trough_row['Date'], color='green', linestyle='--', alpha=0.2)

        # # --- Highlight buy/sell signals to see their relation to peaks/troughs ---
        # if 'Signal' in df.columns:
        #     buys = df.iloc[:i+1][df.iloc[:i+1]['Signal'] == "BUY"]
        #     sells = df.iloc[:i+1][df.iloc[:i+1]['Signal'] == "SELL"]
        #     plt.scatter(buys['Date'], buys[(PRIMARY_INDEX, 'Close')], color='lime', marker='*', s=120, label='Buy Signal' if 'Buy Signal' not in plt.gca().get_legend_handles_labels()[1] else "")
        #     plt.scatter(sells['Date'], sells[(PRIMARY_INDEX, 'Close')], color='red', marker='*', s=120, label='Sell Signal' if 'Sell Signal' not in plt.gca().get_legend_handles_labels()[1] else "")

        # plt.title(f"Day {i+1}: Price, Structure & Positions\n(Peak/Trough lag visualized by vertical lines)")
        # plt.legend(loc='best', fontsize=8, ncol=2)

        # plt.subplot(2,1,2)
        # plt.plot(dates, equity_curve, label='Equity Curve', color='navy')
        # plt.title("Capital Over Time")
        # plt.xlabel("Date")
        # plt.ylabel("Capital")
        # plt.legend()
        # plt.tight_layout()
        # plt.pause(0.01)  # Pause to update the plot

    # ✅ Final exit
    for pos in positions:
        close_price = float(df.iloc[-1][PRIMARY_INDEX]['Close'])
        pnl = (close_price - pos['entry']) * pos['qty']
        charges = ((((pos['total_investment'] - pos['investment']) * 0.15) / 365) * pos['days_held']) + 30
        capital += pos['investment'] + pnl - charges
        trade_log.append({
            'Entry Date': pos['entry_date'],
            'Exit Date': df.iloc[-1]['Date'][''],
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

        # # Print details for closed trades using the pos variable
        # print("-"*50)
        # print(f"TRADE CLOSED on {df.iloc[-1]['Date']['']}")
        # print(f"Entry Date: {pos['entry_date']}")
        # print(f"Entry Price: {pos['entry']:.2f}")
        # print(f"Exit Price: {close_price:.2f}")
        # print(f"Quantity: {pos['qty']}")
        # print(f"Holding Days: {pos['days_held']}")
        # print(f"P&L: {(close_price - pos['entry']) * pos['qty']:.2f} ({((close_price - pos['entry']) * pos['qty'] / pos['total_investment'])*100:.2f}%)")
        # print(f"Charges: {charges:.2f}")
        # print(f"Capital After Trade Close: {capital:.2f}")
        # # print(f"Reason: {'SL' if hit_sl else 'Target' if hit_target else 'Max Days' if max_days else signal}")
        # print("-"*50)

    results = pd.DataFrame(trade_log)
    if not results.empty and 'P&L' in results.columns:
        results['Cumulative P&L'] = results['P&L'].cumsum()
    else:
        results['Cumulative P&L'] = []
    return results, capital

def optimize_parameters(df, stoploss_range, target_range, holding_days_range, stock_name=None):
    # df = df.reset_index()
    results_list = {}
    results_list[stock_name] = []
    for stoploss, target, holding in itertools.product(stoploss_range, target_range, holding_days_range):
        print(f'\nParameters:: Stoploss {stoploss}    :   Target:: {target}      :     Max Holding Period:: {holding}')
        res, final_cap = backtest_swing_strategy(
            df,
            initial_capital=initial_capital,
            stoploss_pct=stoploss,
            target_pct=target,
            max_holding_days=holding
        )
        # print(f'RESULT: {res}')
        # print(f'Final Capital: {final_cap}')
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
            results_list[stock_name].append({
            'stoploss_pct': stoploss,
            'target_pct': target,
            'max_holding_days': holding,
            'final_capital': f'{final_cap:,.2f}',
            'win_rate': f'{win_rate:.2%}',
            'CAGR': f'{cagr:.2%}'
        })
        print(f'CAGR: {cagr:.2%}     WIN Rate: {win_rate:.2%}')
    results_df = pd.DataFrame(results_list[stock_name])
    
    # Save the optimization results for this stock to a CSV file
    if stock_name is not None:
        results_df.to_csv(f'optimization_results_{stock_name}.csv', index=False)
        
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

# --- Function to process a single stock (for multiprocessing) ---
def process_stock(args):
    stock, stoploss_range, target_range, holding_days_range = args
    global PRIMARY_INDEX
    PRIMARY_INDEX = stock
    print(f"\nProcessing {PRIMARY_INDEX}...")
    df = fetch_data()
    if df.empty:
        print(f"Error: No data fetched for {PRIMARY_INDEX}. Skipping.")
        return (PRIMARY_INDEX, None)
    df = calculate_emas(df)
    df = identify_phases(df)
    df = generate_signals(df)
    df = df.reset_index()
    best_params, _ = optimize_parameters(df, stoploss_range, target_range, holding_days_range, stock_name=stock)
    return (PRIMARY_INDEX, best_params.to_dict() if hasattr(best_params, "to_dict") else dict(best_params))

if __name__ == "__main__":
    print("Running Enhanced Dow Theory Analyzer...")

    # List of stock symbols to analyze
    stock_list = input("Enter comma-separated NSE stock names (e.g. TCS,INFY,RELIANCE): ").split(",")
    stock_list = [s.strip().upper() for s in stock_list if s.strip()]

    # Parameter ranges
    stoploss_range = [0.01, 0.02, 0.03]
    target_range = [0.05, 0.075, 0.1]
    holding_days_range = [5, 8]

    # Prepare arguments for each stock
    stock_args = [(stock, stoploss_range, target_range, holding_days_range) for stock in stock_list]

    best_results = {}
    # Use ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=60) as executor:
        results = list(executor.map(process_stock, stock_args))
        for stock, params in results:
            if params is not None:
                best_results[stock] = params

    with open("multi_stock_best_params.json", "w") as f:
        json.dump(best_results, f, indent=2)

    print("\nBest parameters for all stocks saved to multi_stock_best_params.json")

            # # Use best parameters for main backtest
            # results, final_capital = backtest_swing_strategy(df)
            # print(f'RESULT: {results}')
            # print(f"Final capital: ₹{final_capital:,.2f}")


            # # Calculate and print win rate
            # if not results.empty and 'P&L' in results.columns:
            #     num_wins = (results['P&L'] > 0).sum()
            #     num_trades = len(results)
            #     win_rate = num_wins / num_trades if num_trades > 0 else 0
            #     print(f"Win Rate: {win_rate:.2%} ({num_wins}/{num_trades})")
            # else:
            #     print("No trades to calculate win rate.")

            # # Calculate and print CAGR
            # # Calculate number of years from DataFrame date range
            # start_date = pd.to_datetime(df['Date'].iloc[0])
            # end_date = pd.to_datetime(df['Date'].iloc[-1])
            # years = (end_date - start_date).days / 365.25
            # if final_capital > 0 and years > 0:
            #     cagr = (final_capital / initial_capital) ** (1 / years) - 1
            #     print(f"CAGR: {cagr:.2%} per annum")
            # else:
            #     print("CAGR cannot be calculated.")

            
            

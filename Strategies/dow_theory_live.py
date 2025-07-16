import pandas as pd
import numpy as np
from collections import deque

class RealTimeSignalGenerator:
    def __init__(self, symbol, ema_fast=21, ema_slow=55, peak_window=11):
        self.symbol = symbol
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.peak_window = peak_window

        # Internal state
        self.data = pd.DataFrame(columns=['Close'])
        self.ema_fast_vals = []
        self.ema_slow_vals = []
        self.phases = []
        self.hh = []
        self.hl = []
        self.lh = []
        self.ll = []
        self.signals = []

    def update(self, new_price):
        # Append new price
        self.data.loc[len(self.data)] = new_price
        prices = self.data['Close']

        # Calculate EMAs
        ema_fast = prices.ewm(span=self.ema_fast).mean().iloc[-1]
        ema_slow = prices.ewm(span=self.ema_slow).mean().iloc[-1]
        
        self.ema_fast_vals.append(ema_fast)
        self.ema_slow_vals.append(ema_slow)

        # Determine Phase
        if (new_price > ema_slow) and (ema_fast > ema_slow):
            phase = "Bull"
        elif (new_price <= ema_slow) and (ema_fast < ema_slow):
            phase = "Bear"
        else:
            phase = self.phases[-1] if self.phases else "Bear"
        self.phases.append(phase)

        # Peak / Trough detection
        peak = False
        trough = False
        if len(prices) >= self.peak_window:
            window = prices[-self.peak_window:]
            peak = (prices.iloc[-1] == window.max()) and (prices.iloc[-1] != prices.iloc[-2])
            trough = (prices.iloc[-1] == window.min()) and (prices.iloc[-1] != prices.iloc[-2])

        # Trend structure
        if peak and len(prices) > 5:
            if prices.iloc[-1] > prices[-6:-1].max():
                self.hh.append(prices.iloc[-1])
            if prices.iloc[-1] < prices[-6:-1].max():
                self.lh.append(prices.iloc[-1])
        if trough and len(prices) > 5:
            if prices.iloc[-1] > prices[-6:-1].min():
                self.hl.append(prices.iloc[-1])
            if prices.iloc[-1] < prices[-6:-1].min():
                self.ll.append(prices.iloc[-1])

        # Signal logic
        signal = "HOLD"
        if phase == "Bull":
            if self.hh and new_price > self.hh[-1]:
                signal = "BUY"
        self.signals.append("HOLD")
        return signal


generator = RealTimeSignalGenerator("HDFCBANK")

# Simulate real-time LTP feed
for ltp in [1730, 1732, 1733, 1736, 1735, 1738, 1742, 1745, 1748, 1740, 1735]:
    signal = generator.update(ltp)
    print(f"Price: {ltp} → Signal: {signal}")
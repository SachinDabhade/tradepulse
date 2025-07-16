import numpy as np
import pandas as pd
from scipy import stats

def calculate_sma(data, window):
    """Simple Moving Average"""
    return data.rolling(window=window).mean()

def calculate_ema(data, window, alpha=None):
    """Exponential Moving Average"""
    if alpha is None:
        alpha = 2 / (window + 1)
    return data.ewm(alpha=alpha, adjust=False).mean()

def calculate_rsi(data, window=14):
    """Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """MACD (Moving Average Convergence Divergence)"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Bollinger Bands"""
    sma = calculate_sma(data, window)
    std = data.rolling(window=window).std()
    
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    return {
        'upper': upper_band,
        'middle': sma,
        'lower': lower_band
    }

def calculate_stochastic(high, low, close, k_window=14, d_window=3):
    """Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    
    return {
        'k': k_percent,
        'd': d_percent
    }

def calculate_atr(high, low, close, window=14):
    """Average True Range"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=window).mean()
    
    return atr

def calculate_williams_r(high, low, close, window=14):
    """Williams %R"""
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    return williams_r

def calculate_momentum(data, window=10):
    """Price Momentum"""
    return data - data.shift(window)

def calculate_roc(data, window=10):
    """Rate of Change"""
    return ((data - data.shift(window)) / data.shift(window)) * 100

def calculate_cci(high, low, close, window=20):
    """Commodity Channel Index"""
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=window).mean()
    mean_deviation = typical_price.rolling(window=window).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    
    cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
    
    return cci

def calculate_obv(close, volume):
    """On-Balance Volume"""
    direction = np.where(close > close.shift(1), 1, 
                np.where(close < close.shift(1), -1, 0))
    obv = (direction * volume).cumsum()
    
    return obv

def calculate_adx(high, low, close, window=14):
    """Average Directional Index"""
    # True Range
    tr = calculate_atr(high, low, close, 1)
    
    # Directional Movement
    dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low), 
                      np.maximum(high - high.shift(1), 0), 0)
    dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)), 
                       np.maximum(low.shift(1) - low, 0), 0)
    
    # Smoothed versions
    atr_smooth = tr.rolling(window=window).mean()
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=window).mean()
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=window).mean()
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_smooth / atr_smooth)
    di_minus = 100 * (dm_minus_smooth / atr_smooth)
    
    # ADX
    dx = 100 * (np.abs(di_plus - di_minus) / (di_plus + di_minus))
    adx = dx.rolling(window=window).mean()
    
    return {
        'adx': adx,
        'di_plus': di_plus,
        'di_minus': di_minus
    }

def calculate_ichimoku(high, low, close, 
                      conversion_periods=9, 
                      base_periods=26, 
                      lagging_span_periods=52, 
                      displacement=26):
    """Ichimoku Cloud"""
    # Conversion Line (Tenkan-sen)
    conversion_line = (high.rolling(window=conversion_periods).max() + 
                      low.rolling(window=conversion_periods).min()) / 2
    
    # Base Line (Kijun-sen)
    base_line = (high.rolling(window=base_periods).max() + 
                low.rolling(window=base_periods).min()) / 2
    
    # Leading Span A (Senkou Span A)
    leading_span_a = ((conversion_line + base_line) / 2).shift(displacement)
    
    # Leading Span B (Senkou Span B)
    leading_span_b = ((high.rolling(window=lagging_span_periods).max() + 
                      low.rolling(window=lagging_span_periods).min()) / 2).shift(displacement)
    
    # Lagging Span (Chikou Span)
    lagging_span = close.shift(-displacement)
    
    return {
        'conversion_line': conversion_line,
        'base_line': base_line,
        'leading_span_a': leading_span_a,
        'leading_span_b': leading_span_b,
        'lagging_span': lagging_span
    }

def calculate_vwap(high, low, close, volume):
    """Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3
    cumulative_volume = volume.cumsum()
    cumulative_typical_price_volume = (typical_price * volume).cumsum()
    
    vwap = cumulative_typical_price_volume / cumulative_volume
    
    return vwap

def calculate_zscore(data, window=20):
    """Z-Score"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    zscore = (data - rolling_mean) / rolling_std
    
    return zscore

def calculate_fractal_dimension(data, window=20):
    """Fractal Dimension for trend strength"""
    def hurst_exponent(ts):
        """Calculate Hurst Exponent"""
        if len(ts) < 10:
            return 0.5
        
        lags = range(2, min(100, len(ts)//2))
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    
    hurst_values = data.rolling(window=window).apply(
        lambda x: hurst_exponent(x.values), raw=False
    )
    
    return hurst_values

def calculate_technical_signals(data):
    """Calculate multiple technical signals and return combined score"""
    signals = {}
    
    # RSI Signal
    rsi = calculate_rsi(data['close'])
    signals['rsi_oversold'] = (rsi < 30).astype(int)
    signals['rsi_overbought'] = (rsi > 70).astype(int) * -1
    
    # MACD Signal
    macd_data = calculate_macd(data['close'])
    signals['macd_bullish'] = (macd_data['macd'] > macd_data['signal']).astype(int)
    signals['macd_bearish'] = (macd_data['macd'] < macd_data['signal']).astype(int) * -1
    
    # Bollinger Bands
    bb_data = calculate_bollinger_bands(data['close'])
    signals['bb_oversold'] = (data['close'] < bb_data['lower']).astype(int)
    signals['bb_overbought'] = (data['close'] > bb_data['upper']).astype(int) * -1
    
    # Combine signals
    total_signal = sum(signals.values())
    
    return {
        'individual_signals': signals,
        'combined_signal': total_signal,
        'signal_strength': np.abs(total_signal)
    }
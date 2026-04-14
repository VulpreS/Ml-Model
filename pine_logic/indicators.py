import numpy as np
import pandas as pd


class Indicators:
    @staticmethod
    def rsi(close, length=27):
        """RSI индикатор"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def ema(series, length):
        """Exponential Moving Average"""
        return series.ewm(span=length, adjust=False).mean()

    @staticmethod
    def macd(close, fast=11, slow=27, signal=22):
        """MACD индикатор"""
        ema_fast = Indicators.ema(close, fast)
        ema_slow = Indicators.ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = Indicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def nadaraya_watson(src, length=18, bandwidth=2.0):
        """
        Nadaraya-Watson Envelope (no-repaint версия)
        Использует только confirmed бары
        """
        nw_values = np.zeros(len(src))
        nw_values[:] = np.nan

        for i in range(length, len(src)):
            weights = np.zeros(length)
            weighted_sum = 0
            weight_sum = 0

            for j in range(length):
                d = j
                w = np.exp(-0.5 * (d / bandwidth) ** 2)
                weights[j] = w
                weighted_sum += w * src.iloc[i - j]
                weight_sum += w

            if weight_sum > 0:
                nw_values[i] = weighted_sum / weight_sum
            else:
                nw_values[i] = src.iloc[i]

        return pd.Series(nw_values, index=src.index)

    @staticmethod
    def bollinger_bands(close, length=20, mult=2.0):
        """Bollinger Bands"""
        mid = close.rolling(window=length).mean()
        std = close.rolling(window=length).std()
        upper = mid + mult * std
        lower = mid - mult * std
        width = (upper - lower) / mid
        return mid, upper, lower, width

    @staticmethod
    def atr(high, low, close, length=14):
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=length).mean()
        return atr

    @staticmethod
    def structure_detection(high, low, lookback=4):
        """Определение структуры (higher_lows / lower_highs)"""
        higher_lows = (low > low.shift(1)) & (low.shift(1) > low.shift(2)) & \
                      (low.shift(2) > low.shift(3)) & (low.shift(3) > low.shift(4))
        lower_highs = (high < high.shift(1)) & (high.shift(1) < high.shift(2)) & \
                      (high.shift(2) < high.shift(3)) & (high.shift(3) < high.shift(4))
        return higher_lows.astype(float), lower_highs.astype(float)
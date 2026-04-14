import numpy as np
import pandas as pd
from .indicators import Indicators


class SignalGenerator:
    def __init__(self, mode='SCALP'):
        self.mode = mode
        self.cooldown_bars_short = 12
        self.cooldown_bars_long = 18
        self.min_score = 3
        self.need_raw_signals = 2
        self.need_filtered_signals = 2

        # Состояние для no-repaint
        self.last_long_bar = None
        self.last_short_bar = None
        self.long_raw_count = 0
        self.short_raw_count = 0
        self.filtered_long_count = 0
        self.filtered_short_count = 0
        self.last_nw_mid = None
        self.last_nw_upper = None
        self.last_nw_lower = None

    def get_htf(self, current_tf_minutes):
        """Определяет HTF для request.security (точное воспроизведение твоей логики)"""
        if current_tf_minutes <= 5:
            return '60'
        elif current_tf_minutes <= 15:
            return '120'
        elif current_tf_minutes <= 60:
            return '240'
        else:
            return 'D'

    def calculate_slope_cons(self, nw_values):
        """Проверяет 4 бара подряд одного направления"""
        if len(nw_values) < 5:
            return 0.0

        slope = nw_values.diff()
        slope_cons = 0.0

        # Проверяем последние 4 бара
        if (slope.iloc[-1] > 0 and slope.iloc[-2] > 0 and
                slope.iloc[-3] > 0 and slope.iloc[-4] > 0):
            slope_cons = 1.0
        elif (slope.iloc[-1] < 0 and slope.iloc[-2] < 0 and
              slope.iloc[-3] < 0 and slope.iloc[-4] < 0):
            slope_cons = 1.0

        return slope_cons

    def calculate_squeeze(self, bb_width, lookback_bars=60):
        """Рассчитывает squeeze (0-1)"""
        if len(bb_width) < lookback_bars:
            return 0.0

        bb_p90 = bb_width.rolling(window=lookback_bars).max()
        squeeze = 1 - (bb_width / bb_p90)
        squeeze = squeeze.clip(0, 1)
        return squeeze.fillna(0)

    def calculate_structure(self, high, low, lookback=4):
        """Определяет структуру рынка"""
        higher_lows = all(low.iloc[-i] > low.iloc[-i - 1] for i in range(1, lookback + 1))
        lower_highs = all(high.iloc[-i] < high.iloc[-i - 1] for i in range(1, lookback + 1))
        return 1.0 if (higher_lows or lower_highs) else 0.0

    def calculate_score_short_scalp(self, df, bb_width, squeeze, atr, nw_slope):
        """Score для SCALP режима (локальный TF)"""
        # slope_cons
        slope_cons = self.calculate_slope_cons(nw_slope)

        # near_edge (цена у границ BB)
        bb_mid = df['bb_mid'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        close = df['close'].iloc[-1]

        near_edge = 1.0 if (close > bb_upper * 0.98 or close < bb_lower * 1.02) else 0.0

        # ATR expansion
        atr_exp = 1.0 if (atr.iloc[-1] > atr.iloc[-2] and
                          atr.iloc[-2] > atr.iloc[-3] and
                          atr.iloc[-3] > atr.iloc[-4]) else 0.0

        # Structure
        structure = self.calculate_structure(df['high'], df['low'])

        # Squeeze (с уменьшенным lookback для SCALP)
        squeeze_scalp = self.calculate_squeeze(bb_width, lookback_bars=30)

        score = slope_cons + near_edge + atr_exp + structure + squeeze_scalp
        return min(score, 5.0)

    def calculate_bias_scalp(self, df, nw_slope):
        """Определяет bias для SCALP режима"""
        close = df['close'].iloc[-1]
        bb_mid = df['bb_mid'].iloc[-1]
        slope = nw_slope.iloc[-1]

        if slope > 0 and close > bb_mid:
            return "UP"
        elif slope < 0 and close < bb_mid:
            return "DOWN"
        else:
            return "NEUTRAL"

    def generate_signals_for_timeframe(self, df, current_tf_minutes, htf_data=None):
        """
        Генерирует сигналы для одного таймфрейма
        df: DataFrame с OHLCV данными для текущего TF
        htf_data: DataFrame для старшего TF (опционально)
        """
        if len(df) < 50:  # Нужно минимум данных
            return df

        # Копируем чтобы не изменять оригинал
        df = df.copy()

        # === Базовые индикаторы ===
        # RSI (27) на SMA(8)
        ma_source = df['close'].rolling(window=8).mean()
        df['rsi'] = Indicators.rsi(ma_source, length=27)
        df['rsi_ema'] = Indicators.ema(df['rsi'], length=18)

        # MACD (11,27,22)
        macd_line, signal_line, histogram = Indicators.macd(df['close'], 11, 27, 22)
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = histogram
        df['impulse_growing'] = (abs(histogram) > abs(histogram.shift(1))).astype(int)

        # Nadaraya-Watson Envelope (no-repaint!)
        ma6 = df['close'].rolling(window=18).mean()
        nw_mid = Indicators.nadaraya_watson(ma6, length=18, bandwidth=2.0)

        # Envelopes
        env_pct = 1.0
        nw_upper = nw_mid * (1 + env_pct / 100)
        nw_lower = nw_mid * (1 - env_pct / 100)

        df['nw_mid'] = nw_mid
        df['nw_upper'] = nw_upper
        df['nw_lower'] = nw_lower

        # Bollinger Bands
        bb_mid, bb_upper, bb_lower, bb_width = Indicators.bollinger_bands(df['close'], 20, 2.0)
        df['bb_mid'] = bb_mid
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['bb_width'] = bb_width

        # ATR
        df['atr'] = Indicators.atr(df['high'], df['low'], df['close'], 14)
        df['atr_exp'] = ((df['atr'] > df['atr'].shift(1)) &
                         (df['atr'].shift(1) > df['atr'].shift(2)) &
                         (df['atr'].shift(2) > df['atr'].shift(3))).astype(int)

        # === SCALP логика (локальный TF) ===
        nw_slope = df['nw_mid'] - df['nw_mid'].shift(1)

        # Рассчитываем score_short для каждой свечи
        df['slope_cons'] = 0.0
        df['structure'] = 0.0
        df['near_edge'] = 0.0
        df['squeeze'] = 0.0
        df['score_short'] = 0.0
        df['bias_short'] = 'NEUTRAL'

        for i in range(50, len(df)):
            subset = df.iloc[:i + 1]

            # slope_cons
            if i >= 4:
                slopes = nw_slope.iloc[i - 3:i + 1]
                if all(slopes > 0) or all(slopes < 0):
                    df.loc[df.index[i], 'slope_cons'] = 1.0

            # structure
            if i >= 4:
                higher_lows = all(df['low'].iloc[i - j] > df['low'].iloc[i - j - 1] for j in range(4))
                lower_highs = all(df['high'].iloc[i - j] < df['high'].iloc[i - j - 1] for j in range(4))
                if higher_lows or lower_highs:
                    df.loc[df.index[i], 'structure'] = 1.0

            # near_edge
            close_val = df['close'].iloc[i]
            bb_up = df['bb_upper'].iloc[i]
            bb_lo = df['bb_lower'].iloc[i]
            if close_val > bb_up * 0.98 or close_val < bb_lo * 1.02:
                df.loc[df.index[i], 'near_edge'] = 1.0

            # squeeze
            if i >= 30:
                bb_p90 = df['bb_width'].iloc[i - 30:i + 1].max()
                if bb_p90 > 0:
                    squeeze_val = max(0, min(1, 1 - df['bb_width'].iloc[i] / bb_p90))
                    df.loc[df.index[i], 'squeeze'] = squeeze_val

            # score_short
            score = (df['slope_cons'].iloc[i] + df['near_edge'].iloc[i] +
                     df['atr_exp'].iloc[i] + df['structure'].iloc[i] + df['squeeze'].iloc[i])
            df.loc[df.index[i], 'score_short'] = min(score, 5.0)

            # bias_short
            if nw_slope.iloc[i] > 0 and close_val > df['bb_mid'].iloc[i]:
                df.loc[df.index[i], 'bias_short'] = "UP"
            elif nw_slope.iloc[i] < 0 and close_val < df['bb_mid'].iloc[i]:
                df.loc[df.index[i], 'bias_short'] = "DOWN"

        # === HTF данные (если есть) ===
        if htf_data is not None and len(htf_data) > 0:
            # Маппинг HTF данных на текущий TF
            df['score_long'] = np.nan
            df['bias_long'] = 'NEUTRAL'

            for i in range(len(df)):
                current_time = df.index[i]
                # Находим ближайшую HTF свечу
                htf_row = htf_data[htf_data.index <= current_time].iloc[-1] if len(
                    htf_data[htf_data.index <= current_time]) > 0 else None
                if htf_row is not None:
                    df.loc[df.index[i], 'score_long'] = htf_row.get('score_short', 0)
                    df.loc[df.index[i], 'bias_long'] = htf_row.get('bias_short', 'NEUTRAL')
        else:
            # Если HTF нет, используем тот же TF (как fallback)
            df['score_long'] = df['score_short']
            df['bias_long'] = df['bias_short']

        # === Composite Score ===
        df['composite'] = 0.6 * df['score_short'] + 0.4 * df['score_long']

        # Align и Conflict
        align_mask = (df['bias_short'] != 'NEUTRAL') & (df['bias_short'] == df['bias_long'])
        conflict_mask = (df['bias_short'] != 'NEUTRAL') & (df['bias_long'] != 'NEUTRAL') & (
                    df['bias_short'] != df['bias_long'])

        df['composite'] = df['composite'] + align_mask.astype(float) * 0.3 - conflict_mask.astype(float) * 0.5
        df['composite'] = df['composite'].clip(0, 5)

        # === Regime Classification ===
        conditions = [
            (df['score_short'] >= 2.8) & (df['score_long'] >= 2.6) & (df['bias_short'] == df['bias_long']) & (
                        df['bias_short'] != 'NEUTRAL'),
            (df['score_short'] >= 2.8) & (df['score_long'] < 2.6),
            (df['score_long'] >= 2.6) & (df['score_short'] < 2.8),
            conflict_mask,
            (df['score_short'] < 2.0) & (df['score_long'] < 2.0)
        ]
        choices = ['MATURE TREND', 'FRESH MOVE', 'AGING TREND', 'CONFLICT', 'SLEEPING']
        df['regime'] = np.select(conditions, choices, default='NEUTRAL')

        # === Signal Generation (с no-repaint и cooldown) ===
        df['long_signal'] = 0
        df['short_signal'] = 0

        # Сброс счетчиков при новом баре
        long_raw_count = 0
        short_raw_count = 0
        filtered_long_count = 0
        filtered_short_count = 0
        last_long_bar = None
        last_short_bar = None

        for i in range(len(df)):
            if i < 50:
                continue

            # RSI crosses
            rsi_cross_long = (df['rsi'].iloc[i - 2] < df['rsi_ema'].iloc[i - 2] and
                              df['rsi'].iloc[i - 1] > df['rsi_ema'].iloc[i - 1])
            rsi_cross_short = (df['rsi'].iloc[i - 2] > df['rsi_ema'].iloc[i - 2] and
                               df['rsi'].iloc[i - 1] < df['rsi_ema'].iloc[i - 1])

            # MACD scores
            hist_growing = df['impulse_growing'].iloc[i]
            hist_positive = df['macd_hist'].iloc[i] > 0

            rsi_score_long = 2.0 if df['rsi'].iloc[i] < 27 else -0.5
            rsi_score_short = 2.0 if df['rsi'].iloc[i] > 72 else -0.5

            macd_score_long = 1.0 if hist_growing else -1.0
            macd_score_short = 1.0 if not hist_growing else -1.0

            early_score_long = rsi_score_long + macd_score_long
            early_score_short = rsi_score_short + macd_score_short

            # Raw signals
            can_long = (last_long_bar is None or i - last_long_bar >= self.cooldown_bars_short)
            can_short = (last_short_bar is None or i - last_short_bar >= self.cooldown_bars_short)

            raw_long = early_score_long >= self.min_score and can_long
            raw_short = early_score_short >= self.min_score and can_short

            long_raw_count = long_raw_count + 1 if raw_long else 0
            short_raw_count = short_raw_count + 1 if raw_short else 0

            filtered_long = long_raw_count >= self.need_raw_signals + 1
            filtered_short = short_raw_count >= self.need_raw_signals

            filtered_long_count = filtered_long_count + 1 if filtered_long else 0
            filtered_short_count = filtered_short_count + 1 if filtered_short else 0

            confirmed_long = filtered_long_count >= self.need_filtered_signals
            confirmed_short = filtered_short_count >= self.need_filtered_signals

            # Cooldown для confirmed сигналов
            can_long_final = (last_long_bar is None or i - last_long_bar >= self.cooldown_bars_long)
            can_short_final = (last_short_bar is None or i - last_short_bar >= self.cooldown_bars_short)

            new_long = confirmed_long and can_long_final
            new_short = confirmed_short and can_short_final

            if new_long:
                df.loc[df.index[i], 'long_signal'] = 1
                last_long_bar = i
                last_short_bar = None

            if new_short:
                df.loc[df.index[i], 'short_signal'] = 1
                last_short_bar = i
                last_long_bar = None

        return df
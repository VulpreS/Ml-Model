import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import hashlib


class PatternLearner:
    def __init__(self, db_manager):
        self.db = db_manager

    def learn_from_history(self, symbols: List[str], timeframes: List[str]):
        """Анализирует исторические паттерны и сохраняет статистику"""
        patterns = {}

        for symbol in symbols:
            for tf in timeframes:
                print(f"Learning patterns for {symbol} {tf}...")

                # Загружаем данные с сигналами
                df = self.db.load_ohlcv(symbol, tf)

                if df.empty or len(df) < 100:
                    continue

                # Анализируем каждый сигнал
                signals_df = df[df['long_signal'] == 1]

                for idx, signal in signals_df.iterrows():
                    signal_time = pd.to_datetime(signal['timestamp'], unit='ms')

                    # Смотрим движение через 1h, 4h, 24h
                    for lookahead in ['1h', '4h', '24h']:
                        pattern_key = self._create_pattern_key(
                            signal['regime'],
                            self._composite_to_range(signal['composite']),
                            signal['bias_short'],
                            signal['bias_long']
                        )

                        # Рассчитываем движение цены
                        future_price = self._get_future_price(df, idx, lookahead)
                        if future_price:
                            price_change = (future_price - signal['close']) / signal['close'] * 100

                            if pattern_key not in patterns:
                                patterns[pattern_key] = {
                                    'movements': [],
                                    'wins': 0,
                                    'total': 0,
                                    'lookahead': lookahead
                                }

                            patterns[pattern_key]['movements'].append(price_change)
                            patterns[pattern_key]['total'] += 1

                            if price_change > 0:  # Для long сигналов
                                patterns[pattern_key]['wins'] += 1

        # Сохраняем результаты в БД
        self._save_patterns(patterns)

    def _create_pattern_key(self, regime: str, composite_range: str, bias_short: str, bias_long: str) -> str:
        """Создает уникальный ключ для паттерна"""
        pattern_str = f"{regime}|{composite_range}|{bias_short}|{bias_long}"
        return hashlib.md5(pattern_str.encode()).hexdigest()

    def _composite_to_range(self, composite: float) -> str:
        """Конвертирует composite score в диапазон"""
        if composite < 2.5:
            return "LOW"
        elif composite < 3.2:
            return "MEDIUM"
        else:
            return "HIGH"

    def _get_future_price(self, df: pd.DataFrame, current_idx: int, lookahead: str) -> float:
        """Получает будущую цену через указанный период"""
        current_time = pd.to_datetime(df.iloc[current_idx]['timestamp'], unit='ms')

        # Конвертируем lookahead в timedelta
        if lookahead == '1h':
            delta = timedelta(hours=1)
        elif lookahead == '4h':
            delta = timedelta(hours=4)
        else:  # 24h
            delta = timedelta(hours=24)

        target_time = current_time + delta

        # Ищем ближайшую свечу после target_time
        future_rows = df[df['timestamp'] > target_time.timestamp() * 1000]

        if not future_rows.empty:
            return future_rows.iloc[0]['close']

        return None

    def _save_patterns(self, patterns: Dict):
        """Сохраняет паттерны в БД"""
        now = int(datetime.now().timestamp() * 1000)

        for pattern_key, data in patterns.items():
            if data['total'] > 0:
                probability = data['wins'] / data['total']
                expected_move = np.mean(data['movements'])

                self.db.save_pattern(
                    pattern_key,
                    probability,
                    expected_move,
                    data['wins'] / data['total'],
                    data['total'],
                    now
                )

    def get_prediction(self, regime: str, composite: float, bias_short: str, bias_long: str, lookahead: str = '4h') -> \
    Tuple[float, float]:
        """Получает предсказание для текущего паттерна"""
        pattern_key = self._create_pattern_key(
            regime,
            self._composite_to_range(composite),
            bias_short,
            bias_long
        )

        # Загружаем паттерн из БД
        pattern = self.db.load_pattern(pattern_key, lookahead)

        if pattern:
            return pattern['probability'], pattern['expected_move_pct']
        else:
            # Дефолтные значения если паттерн не найден
            return 0.5, 0.5
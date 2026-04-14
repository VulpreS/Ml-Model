import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier


class MLModel:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        self.is_trained = False


    def build_features(self, state: dict):
        features = []

        for tf in sorted(state.keys()):

            s = state[tf]

            # === CORE ===
            features.extend([
                s.get('composite', 0) / 5.0,
                s.get('score_short', 0) / 5.0,
                s.get('score_long', 0) / 5.0,
            ])

            # === RSI ===
            features.extend([
                s.get('rsi', 50) / 100,
                s.get('rsi_ema', 50) / 100,
                (s.get('rsi', 50) - s.get('rsi_ema', 50)) / 100
            ])

            # === MACD ===
            features.extend([
                s.get('macd', 0),
                s.get('macd_signal', 0),
                s.get('macd_hist', 0),
            ])

            # === BOLLINGER ===
            bb_mid = s.get('bb_mid', 0)
            bb_width = s.get('bb_width', 1)

            features.extend([
                bb_width,
                (s.get('close', 0) - bb_mid) / (bb_width + 1e-6)
            ])

            # === NADARAYA-WATSON ===
            nw_mid = s.get('nw_mid', 0)

            features.extend([
                (s.get('close', 0) - nw_mid) / (nw_mid + 1e-6),
                (s.get('nw_upper', 0) - s.get('nw_lower', 0)) / (nw_mid + 1e-6)
            ])

            # === VOLATILITY ===
            features.extend([
                s.get('atr', 0) / (s.get('close', 1) + 1e-6),
                s.get('atr_exp', 0)/(s.get('close', 1) + 1e-6),
            ])

            # === STRUCTURE ===
            features.extend([
                s.get('structure', 0),
                s.get('slope_cons', 0),
                s.get('near_edge', 0)
            ])

            # === SQUEEZE ===
            features.append(s.get('squeeze', 0))
            # momentum
            features.append(
                (s.get('close', 0) - s.get('nw_mid', 0)) / (s.get('nw_mid', 1) + 1e-6)
            )
            # === BIAS ===
            features.extend([
                1 if s.get('bias_short') == 'UP' else -1 if s.get('bias_short') == 'DOWN' else 0,
                1 if s.get('bias_long') == 'UP' else -1 if s.get('bias_long') == 'DOWN' else 0,
            ])

            # === REGIME (one-hot) ===
            regime = s.get('regime', '')

            features.extend([
                1 if regime == 'MATURE TREND' else 0,
                1 if regime == 'FRESH MOVE' else 0,
                1 if regime == 'AGING TREND' else 0,
                1 if regime == 'CONFLICT' else 0,
                1 if regime == 'SLEEPING' else 0,
            ])

            # === META FEATURES ===
            features.extend([
                1 if s.get('bias_short') == s.get('bias_long') else 0,
                abs(s.get('score_short', 0) - s.get('score_long', 0)) / 5.0
            ])

        return np.array(features)

    def train(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, state):
        if not self.is_trained:
            return {
                'direction': 'NEUTRAL',
                'probability': 0.5,
                'expected_move': 0.0,
                'tp_pct': 1.0,
                'sl_pct': 1.0
            }

        x = self.build_features(state).reshape(1, -1)

        prob = self.model.predict_proba(x)[0][1]

        if prob > 0.6:
            direction = 'LONG'
        elif prob < 0.4:
            direction = 'SHORT'
        else:
            direction = 'NEUTRAL'

        return {
            'direction': direction,
            'probability': float(prob),
            'expected_move': float(abs(prob - 0.5) * 4),  # усиливаем уверенность
            'tp_pct': 1.2 + prob,
            'sl_pct': 0.8
        }

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        try:
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
                self.is_trained = True
        except:
            self.is_trained = False
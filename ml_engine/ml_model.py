import pickle

import numpy as np

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


class MLModel:
    def __init__(self):
        self.model = None
        if XGBClassifier is not None:
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
            )
        self.is_trained = False

    def _encode_bias(self, value):
        if value in {"UP", "STRONG"}:
            return 1
        if value in {"DOWN", "WEAK"}:
            return -1
        return 0

    def build_features(self, state):
        features = []

        for tf in sorted(state.keys()):
            s = state[tf]
            bb_mid = s.get("bb_mid", 0.0)
            bb_width = s.get("bb_width", 1.0)
            nw_mid = s.get("nw_mid", 0.0)
            close = s.get("close", 0.0)
            regime = s.get("regime", "")

            features.extend(
                [
                    s.get("composite", 0) / 5.0,
                    s.get("score_short", 0) / 5.0,
                    s.get("score_long", 0) / 5.0,
                    s.get("rsi", 50) / 100.0,
                    s.get("rsi_ema", 50) / 100.0,
                    (s.get("rsi", 50) - s.get("rsi_ema", 50)) / 100.0,
                    s.get("macd", 0.0),
                    s.get("macd_signal", 0.0),
                    s.get("macd_hist", 0.0),
                    bb_width,
                    (close - bb_mid) / (bb_width + 1e-6),
                    (close - nw_mid) / (abs(nw_mid) + 1e-6),
                    (s.get("nw_upper", 0.0) - s.get("nw_lower", 0.0)) / (abs(nw_mid) + 1e-6),
                    s.get("atr", 0.0) / (close + 1e-6),
                    s.get("atr_exp", 0.0) / (close + 1e-6),
                    s.get("structure", 0.0),
                    s.get("slope_cons", 0.0),
                    s.get("near_edge", 0.0),
                    s.get("squeeze", 0.0),
                    self._encode_bias(s.get("bias_short")),
                    self._encode_bias(s.get("bias_long")),
                    1 if regime == "MATURE TREND" else 0,
                    1 if regime == "FRESH MOVE" else 0,
                    1 if regime == "AGING TREND" else 0,
                    1 if regime == "CONFLICT" else 0,
                    1 if regime == "SLEEPING" else 0,
                    1 if s.get("bias_short") == s.get("bias_long") else 0,
                    abs(s.get("score_short", 0) - s.get("score_long", 0)) / 5.0,
                ]
            )

        return np.array(features, dtype=float)

    def train(self, X, y):
        if self.model is None:
            print("[MLModel] XGBoost is not available, training skipped")
            self.is_trained = False
            return
        print("[MLModel] Starting XGBoost.fit()")
        self.model.fit(X, y)
        print("[MLModel] XGBoost.fit() completed")
        self.is_trained = True

    def predict(self, state):
        if not self.is_trained or self.model is None:
            return {
                "direction": "NEUTRAL",
                "probability": 0.5,
                "expected_move": 0.0,
                "tp_pct": 1.0,
                "sl_pct": 1.0,
            }

        x = self.build_features(state).reshape(1, -1)
        prob = float(self.model.predict_proba(x)[0][1])

        if prob > 0.6:
            direction = "LONG"
        elif prob < 0.4:
            direction = "SHORT"
        else:
            direction = "NEUTRAL"

        return {
            "direction": direction,
            "probability": prob,
            "expected_move": float(abs(prob - 0.5) * 4),
            "tp_pct": 1.2 + prob,
            "sl_pct": 0.8,
        }

    def save(self, path):
        if self.model is None:
            return
        with open(path, "wb") as file_obj:
            pickle.dump(self.model, file_obj)

    def load(self, path):
        try:
            with open(path, "rb") as file_obj:
                self.model = pickle.load(file_obj)
                self.is_trained = True
        except (FileNotFoundError, pickle.UnpicklingError, EOFError):
            self.is_trained = False

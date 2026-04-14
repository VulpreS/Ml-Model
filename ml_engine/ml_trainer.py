from collections import Counter
from time import perf_counter

import numpy as np

from config import ML_MODEL_PATH
from ml_engine.ml_model import MLModel


class MLTrainer:
    def __init__(self, db, symbols, timeframes):
        self.db = db
        self.symbols = symbols
        self.timeframes = timeframes
        self.model = MLModel()

    def _log(self, message):
        print(f"[MLTrainer] {message}")

    def _load_symbol_frames(self, symbol):
        frames = {}

        for timeframe in self.timeframes:
            df = self.db.load_ohlcv(symbol, timeframe)
            if df.empty:
                self._log(f"{symbol} {timeframe}: no data loaded")
                continue

            frame = df.sort_values("timestamp").reset_index(drop=True).copy()
            frame["timestamp"] = frame["timestamp"].astype("int64")
            frames[timeframe] = frame
            self._log(f"{symbol} {timeframe}: cached {len(frame)} rows")

        return frames

    def _build_mtf_state_from_frames(self, frames, timestamp):
        state = {}

        for timeframe, df in frames.items():
            df_filtered = df[df["timestamp"] < timestamp]
            if df_filtered.empty:
                continue

            row = df_filtered.iloc[-1]
            state[timeframe] = {
                "close": row["close"],
                "rsi": row["rsi"],
                "rsi_ema": row["rsi_ema"],
                "macd": row["macd"],
                "macd_signal": row["macd_signal"],
                "macd_hist": row["macd_hist"],
                "bb_mid": row["bb_mid"],
                "bb_width": row["bb_width"],
                "nw_mid": row["nw_mid"],
                "nw_upper": row["nw_upper"],
                "nw_lower": row["nw_lower"],
                "atr": row["atr"],
                "atr_exp": row["atr_exp"],
                "structure": row["structure"],
                "slope_cons": row["slope_cons"],
                "near_edge": row["near_edge"],
                "squeeze": row["squeeze"],
                "score_short": row["score_short"],
                "score_long": row["score_long"],
                "bias_short": row["bias_short"],
                "bias_long": row["bias_long"],
                "composite": row["composite"],
                "regime": row["regime"],
            }

        return state

    def build_mtf_state(self, symbol, timestamp):
        frames = self._load_symbol_frames(symbol)
        return self._build_mtf_state_from_frames(frames, int(timestamp))

    def build_dataset(self):
        build_started_at = perf_counter()
        X = []
        y = []
        main_tf = next(iter(self.timeframes.keys()))

        self._log(f"Building dataset from main timeframe {main_tf}")

        for symbol in self.symbols:
            symbol_started_at = perf_counter()
            frames = self._load_symbol_frames(symbol)
            df = frames.get(main_tf)

            if df is None or df.empty:
                self._log(f"{symbol}: skipped because no data in {main_tf}")
                continue

            if len(df) < 200:
                self._log(f"{symbol}: skipped because only {len(df)} rows in {main_tf}")
                continue

            added_before = len(X)
            self._log(f"{symbol}: processing {len(df)} rows from {main_tf}")

            for index in range(len(df) - 20):
                row = df.iloc[index]
                timestamp = int(row["timestamp"])
                state = self._build_mtf_state_from_frames(frames, timestamp)

                if len(state) < len(self.timeframes):
                    continue

                current_price = row["close"]
                future_price = df.iloc[index + 10]["close"]
                future_price_far = df.iloc[index + 20]["close"]
                move = abs((future_price_far - current_price) / current_price)

                if move < 0.003:
                    continue

                target = 1 if future_price > current_price else 0
                X.append(self.model.build_features(state))
                y.append(target)

            added_now = len(X) - added_before
            self._log(
                f"{symbol}: added {added_now} samples in {perf_counter() - symbol_started_at:.2f}s"
            )

        self._log(
            f"Dataset build complete: samples={len(X)} in {perf_counter() - build_started_at:.2f}s"
        )
        return np.array(X), np.array(y)

    def train(self):
        train_started_at = perf_counter()
        X, y = self.build_dataset()

        if len(X) < 50:
            self._log("Not enough data for ML")
            return None

        self._log(
            f"Starting fit: X_shape={X.shape}, y_shape={y.shape}, class_balance={dict(Counter(y))}"
        )
        fit_started_at = perf_counter()
        self.model.train(X, y)
        self._log(f"Fit complete in {perf_counter() - fit_started_at:.2f}s")

        self.model.save(ML_MODEL_PATH)
        self._log(f"Model saved to {ML_MODEL_PATH}")
        self._log(f"Training finished in {perf_counter() - train_started_at:.2f}s")
        return self.model

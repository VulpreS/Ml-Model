import time
from datetime import datetime, timedelta

import pandas as pd

from .bybit_raw import BybitRawClient


class DataUpdater:
    def __init__(self, db_manager, api_key=None, api_secret=None):
        self.db = db_manager
        self.client = BybitRawClient()

    def _log(self, message):
        print(f"[DataUpdater] {message}")

    def _normalize_timestamps(self, df):
        if df.empty or "timestamp" not in df.columns:
            return df

        normalized = df.copy()
        if pd.api.types.is_numeric_dtype(normalized["timestamp"]):
            normalized["timestamp"] = normalized["timestamp"].astype("int64")
            return normalized

        parsed = pd.to_datetime(normalized["timestamp"], errors="coerce")
        normalized = normalized.loc[parsed.notna()].copy()
        normalized["timestamp"] = (
            parsed.loc[parsed.notna()].astype("int64") // 10**6
        ).astype("int64")
        return normalized

    def update_all_data(self, symbols, timeframes, days_back=730):
        for symbol in symbols:
            for tf in timeframes.keys():
                self._log(f"Loading {symbol} {tf}")
                self.db.create_timeframe_table(symbol, tf)

                symbol_api = symbol.replace("/", "").replace(":USDT", "")
                df = self.client.fetch_all_history(symbol_api, tf, days_back)

                if not df.empty:
                    df = self._normalize_timestamps(df)
                    df = self.add_features(df).dropna().reset_index(drop=True)
                    self.db.save_ohlcv(symbol, tf, df)
                    self._log(f"{symbol} {tf}: saved {len(df)} candles")
                else:
                    self._log(f"{symbol} {tf}: no data from API, generating test data")
                    self._generate_test_data(symbol, tf, days_back)

                time.sleep(0.5)

    def add_features(self, df):
        import numpy as np

        df = df.copy()

        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / (loss + 1e-6)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi_ema"] = df["rsi"].ewm(span=14).mean()

        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        mid = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        df["bb_mid"] = mid
        df["bb_width"] = std * 2

        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = np.maximum(high_low, np.maximum(high_close, low_close))

        df["atr"] = tr.rolling(14).mean()
        df["atr_exp"] = df["atr"].ewm(span=14).mean()

        df["nw_mid"] = df["close"].ewm(span=20).mean()
        df["nw_upper"] = df["nw_mid"] + df["atr"]
        df["nw_lower"] = df["nw_mid"] - df["atr"]

        df["structure"] = np.where(df["close"] > df["nw_mid"], 1, -1)
        df["slope"] = df["nw_mid"].diff()
        df["slope_cons"] = np.where(df["slope"] > 0, 1, -1)

        df["near_edge"] = np.where(
            (df["close"] > df["nw_upper"] * 0.98) |
            (df["close"] < df["nw_lower"] * 1.02),
            1,
            0,
        )

        df["squeeze"] = np.where(df["bb_width"] < df["atr"], 1, 0)

        df["score_long"] = (
            (df["rsi"] > df["rsi_ema"]).astype(int) +
            (df["macd"] > df["macd_signal"]).astype(int) +
            (df["structure"] > 0).astype(int) +
            (df["slope_cons"] > 0).astype(int)
        )

        df["score_short"] = (
            (df["rsi"] < df["rsi_ema"]).astype(int) +
            (df["macd"] < df["macd_signal"]).astype(int) +
            (df["structure"] < 0).astype(int) +
            (df["slope_cons"] < 0).astype(int)
        )

        df["bias_long"] = np.where(df["score_long"] >= 3, "STRONG", "WEAK")
        df["bias_short"] = np.where(df["score_short"] >= 3, "STRONG", "WEAK")
        df["composite"] = df["score_long"] - df["score_short"]
        df["regime"] = np.where(abs(df["composite"]) >= 2, "TREND", "CONFLICT")
        df["long_signal"] = np.where(df["composite"] >= 3, 1, 0)
        df["short_signal"] = np.where(df["composite"] <= -3, 1, 0)

        return df

    def _generate_test_data(self, symbol, tf, days_back):
        import random

        tf_minutes = {
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "1d": 1440,
            "1w": 10080,
        }.get(tf, 60)

        candles_count = (days_back * 1440) // tf_minutes
        candles_count = min(candles_count, 5000)

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)

        timestamps = []
        current_time = start_time
        for _ in range(candles_count):
            timestamps.append(int(current_time.timestamp() * 1000))
            current_time += timedelta(minutes=tf_minutes)

        base_price = 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100
        random.seed(hash(symbol) + hash(tf))

        data = []
        price = base_price
        for ts in timestamps:
            change = random.uniform(-0.02, 0.02)
            price = price * (1 + change)
            data.append(
                {
                    "timestamp": ts,
                    "open": price * (1 + random.uniform(-0.001, 0.001)),
                    "high": price * (1 + random.uniform(0, 0.005)),
                    "low": price * (1 - random.uniform(0, 0.005)),
                    "close": price,
                    "volume": random.uniform(10, 1000),
                }
            )

        df = pd.DataFrame(data)
        df = self._normalize_timestamps(df)
        self.db.save_ohlcv(symbol, tf, df)
        self._log(f"{symbol} {tf}: generated {len(df)} test candles")

    def update_recent_data(self, symbols, timeframes, hours_back=168):
        for symbol in symbols:
            for tf in timeframes.keys():
                self._log(f"Updating {symbol} {tf}")
                symbol_api = symbol.replace("/", "").replace(":USDT", "")
                df_existing = self.db.load_ohlcv(symbol, tf)

                if df_existing.empty:
                    self._log(f"{symbol} {tf}: no local data, starting full load")
                    df_full = self.client.fetch_all_history(symbol_api, tf, 730)
                    if not df_full.empty:
                        df_full = self._normalize_timestamps(df_full)
                        df_full = self.add_features(df_full).dropna().reset_index(drop=True)
                        self.db.save_ohlcv(symbol, tf, df_full)
                        self._log(f"{symbol} {tf}: full load saved {len(df_full)} rows")
                    else:
                        self._log(f"{symbol} {tf}: full load returned no rows")
                    continue

                df_existing = self._normalize_timestamps(df_existing)
                last_ts_ms = int(df_existing["timestamp"].max())

                since_time = datetime.now() - timedelta(hours=hours_back)
                since_ms = int(since_time.timestamp() * 1000)
                start_ms = max(last_ts_ms, since_ms)

                self._log(
                    f"{symbol} {tf}: existing_rows={len(df_existing)}, last_ts_ms={last_ts_ms}, start_ms={start_ms}"
                )

                df_new = self.client.fetch_recent_history(symbol_api, tf, start_ms)
                if df_new.empty:
                    self._log(f"{symbol} {tf}: no new data")
                    continue

                df_new = self._normalize_timestamps(df_new)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined = df_combined.drop_duplicates(subset="timestamp", keep="last")
                df_combined = df_combined.sort_values("timestamp").reset_index(drop=True)
                df_combined = self.add_features(df_combined).dropna().reset_index(drop=True)

                self.db.save_ohlcv(symbol, tf, df_combined)
                self._log(
                    f"{symbol} {tf}: added {len(df_new)} candles, total={len(df_combined)}"
                )
                time.sleep(0.2)

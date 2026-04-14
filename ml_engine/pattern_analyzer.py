import sqlite3
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd


class PatternAnalyzer:
    def __init__(self, db_path):
        self.db_path = db_path
        self.patterns_db = {}

    def _log(self, message):
        print(f"[PatternAnalyzer] {message}")

    def _format_symbol(self, symbol):
        return symbol.replace("/", "_").replace(":", "_")

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

    def learn_from_database(self, symbols, timeframes, min_samples=10):
        self._log("Starting deep pattern learning")
        all_patterns = []

        for symbol in symbols:
            self._log(f"Analyzing {symbol}")
            tf_data = {}

            for tf in timeframes:
                df = self._load_data(symbol, tf)
                if not df.empty:
                    tf_data[tf] = df

            if len(tf_data) < 2:
                self._log(f"{symbol}: not enough data across timeframes")
                continue

            patterns = self._extract_patterns(symbol, tf_data, timeframes, min_samples)
            all_patterns.extend(patterns)
            self._log(f"{symbol}: extracted {len(patterns)} patterns")

        self._train_model(all_patterns)
        self._log(f"Training complete. Found {len(all_patterns)} patterns")
        return self.patterns_db

    def _load_data(self, symbol, timeframe):
        table_name = f"{self._format_symbol(symbol)}_{timeframe}"

        with sqlite3.connect(self.db_path) as conn:
            try:
                df = pd.read_sql_query(
                    f"""
                    SELECT timestamp, close, high, low, volume,
                           composite, regime, bias_short, bias_long,
                           long_signal, short_signal, score_short, score_long
                    FROM {table_name}
                    ORDER BY timestamp ASC
                    """,
                    conn,
                )
            except Exception:
                return pd.DataFrame()

        return self._normalize_timestamps(df)

    def _extract_patterns(self, symbol, tf_data, timeframes, min_samples):
        patterns = []
        main_tf = min(timeframes.keys(), key=lambda key: timeframes[key])
        main_df = tf_data[main_tf]

        for i in range(len(main_df) - 100):
            current_time = int(main_df.iloc[i]["timestamp"])
            state = self._get_mtf_state(tf_data, current_time)
            if not state:
                continue

            for lookahead in ("1h", "4h", "24h"):
                future_price = self._get_future_price(main_df, i, lookahead, timeframes[main_tf])
                if future_price is None:
                    continue

                current_price = main_df.iloc[i]["close"]
                price_change = (future_price - current_price) / current_price * 100
                patterns.append(
                    {
                        "timestamp": current_time,
                        "symbol": symbol,
                        "main_tf": main_tf,
                        "state": state,
                        "price_change": price_change,
                        "is_profit": price_change > 0,
                        "lookahead": lookahead,
                        "entry_price": current_price,
                        "future_price": future_price,
                    }
                )

        if len(patterns) < min_samples:
            return []
        return patterns

    def _get_mtf_state(self, tf_data, timestamp):
        state = {}

        for tf, df in tf_data.items():
            mask = df["timestamp"] <= timestamp
            if not mask.any():
                continue

            closest = df.loc[mask].iloc[-1]
            state[tf] = {
                "regime": closest["regime"],
                "bias_short": closest["bias_short"],
                "bias_long": closest["bias_long"],
                "composite": closest["composite"],
                "score_short": closest["score_short"],
                "score_long": closest["score_long"],
                "long_signal": int(closest.get("long_signal", 0)),
                "short_signal": int(closest.get("short_signal", 0)),
            }

        return state if len(state) == len(tf_data) else None

    def _get_future_price(self, df, current_idx, lookahead, tf_minutes):
        lookahead_minutes = {"1h": 60, "4h": 240, "24h": 1440}.get(lookahead, 60)
        bars_ahead = lookahead_minutes // tf_minutes
        future_idx = current_idx + bars_ahead
        if future_idx < len(df):
            return df.iloc[future_idx]["close"]
        return None

    def _train_model(self, patterns):
        self._log("Training model on patterns")
        pattern_groups = defaultdict(list)

        for pattern in patterns:
            key = self._create_pattern_key(pattern["state"])
            pattern_groups[key].append(pattern)

        for key, group in pattern_groups.items():
            if len(group) < 5:
                continue

            for lookahead in ("1h", "4h", "24h"):
                lookahead_patterns = [p for p in group if p["lookahead"] == lookahead]
                if len(lookahead_patterns) < 3:
                    continue

                price_changes = [p["price_change"] for p in lookahead_patterns]
                wins = [p for p in lookahead_patterns if p["is_profit"]]
                avg_change = np.mean(price_changes)
                direction = "LONG" if avg_change > 0 else "SHORT" if avg_change < 0 else "NEUTRAL"
                probability = len(wins) / len(lookahead_patterns)
                expected_move = abs(avg_change)

                self.patterns_db[key] = {
                    "direction": direction,
                    "probability": probability,
                    "expected_move": expected_move,
                    "tp_pct": expected_move * 1.5,
                    "sl_pct": expected_move * 0.75,
                    "samples": len(lookahead_patterns),
                    "win_rate": probability,
                    "avg_change": avg_change,
                    "lookahead": lookahead,
                }

        self._save_model()

    def _create_pattern_key(self, state):
        key_parts = []

        for tf in sorted(state.keys()):
            tf_state = state[tf]
            regime = tf_state["regime"]
            bias = f"{tf_state['bias_short']}/{tf_state['bias_long']}"
            composite = round(tf_state["composite"], 1)

            if composite < 2.5:
                comp_range = "LOW"
            elif composite < 3.5:
                comp_range = "MED"
            else:
                comp_range = "HIGH"

            key_parts.append(f"{tf}:{regime}|{bias}|{comp_range}")

        return " | ".join(key_parts)

    def _save_model(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trained_patterns (
                    pattern_key TEXT PRIMARY KEY,
                    direction TEXT,
                    probability REAL,
                    expected_move REAL,
                    tp_pct REAL,
                    sl_pct REAL,
                    samples INTEGER,
                    win_rate REAL,
                    lookahead TEXT,
                    created_at INTEGER
                )
                """
            )

            for key, data in self.patterns_db.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO trained_patterns
                    (pattern_key, direction, probability, expected_move, tp_pct, sl_pct, samples, win_rate, lookahead, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key,
                        data["direction"],
                        data["probability"],
                        data["expected_move"],
                        data["tp_pct"],
                        data["sl_pct"],
                        data["samples"],
                        data["win_rate"],
                        data["lookahead"],
                        int(datetime.now().timestamp()),
                    ),
                )

        self._log(f"Saved {len(self.patterns_db)} patterns to database")

    def predict(self, current_state, lookahead="4h"):
        key = self._create_pattern_key(current_state)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT * FROM trained_patterns
                WHERE pattern_key = ? AND lookahead = ?
                ORDER BY samples DESC
                LIMIT 1
                """,
                conn,
                params=(key, lookahead),
            )

        if not df.empty:
            row = df.iloc[0]
            return {
                "direction": row["direction"],
                "probability": row["probability"],
                "expected_move": row["expected_move"],
                "tp_pct": row["tp_pct"],
                "sl_pct": row["sl_pct"],
                "samples": row["samples"],
            }

        return {
            "direction": "NEUTRAL",
            "probability": 0.5,
            "expected_move": 0.5,
            "tp_pct": 0.75,
            "sl_pct": 0.5,
            "samples": 0,
        }

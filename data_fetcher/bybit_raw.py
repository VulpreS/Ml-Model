import time
from datetime import datetime, timedelta

import pandas as pd
import requests


class BybitRawClient:
    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        self.session = requests.Session()

    def _log(self, message):
        print(f"[BybitRaw] {message}")

    def _fetch_klines(self, symbol, interval, start, limit=1000):
        url = f"{self.BASE_URL}/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "start": start,
            "limit": limit,
        }

        while True:
            try:
                response = self.session.get(url, params=params, timeout=10)
                data = response.json()

                if data["retCode"] == 0:
                    return data["result"]["list"]

                if data["retCode"] == 10006:
                    self._log(f"Rate limit for {symbol} {interval}, sleeping 1s")
                    time.sleep(1)
                    continue

                self._log(f"API error for {symbol} {interval}: {data}")
                return []
            except Exception as exc:
                self._log(f"Network error for {symbol} {interval}: {exc}")
                time.sleep(1)

    def fetch_recent_history(self, symbol, timeframe, since_ms):
        interval_map = {"15m": "15", "1h": "60", "4h": "240"}
        interval = interval_map.get(timeframe, "60")
        all_data = []

        while True:
            data = self._fetch_klines(symbol, interval, since_ms)
            if not data:
                break

            data = data[::-1]
            parsed = [
                [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])]
                for row in data
            ]
            all_data.extend(parsed)
            self._log(f"{symbol} {timeframe}: fetched {len(parsed)} new candles, total={len(all_data)}")

            if len(data) < 1000:
                break

            since_ms = data[-1][0] + 1
            time.sleep(0.1)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.sort_values("timestamp").reset_index(drop=True)

    def fetch_all_history(self, symbol, timeframe, days_back=730):
        interval_map = {"15m": "15", "1h": "60", "4h": "240"}
        interval = interval_map[timeframe]
        start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
        all_data = []

        while True:
            data = self._fetch_klines(symbol, interval, start_time)
            if not data:
                break

            data = data[::-1]
            parsed = [
                [int(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])]
                for row in data
            ]
            all_data.extend(parsed)

            last_ts = parsed[-1][0]
            new_start = last_ts + 1
            self._log(f"{symbol} {timeframe}: fetched {len(parsed)} candles, total={len(all_data)}")

            if new_start <= start_time:
                self._log(f"{symbol} {timeframe}: no progress, stopping")
                break

            start_time = new_start
            if len(parsed) < 1000:
                break

            time.sleep(0.1)

        df = pd.DataFrame(
            all_data,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

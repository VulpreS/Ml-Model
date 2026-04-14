import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta


class BybitClient:
    def __init__(self, api_key=None, api_secret=None, testnet=False):
        self.exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            "rateLimit": 500,
            'options': {'defaultType': 'linear'}
        })
        if testnet:
            self.exchange.set_sandbox_mode(False)

    def fetch_all_history_robust(self, symbol, timeframe, days_back=730):
        tf_minutes = self._timeframe_to_minutes(timeframe)
        max_limit = 1000

        start_date = datetime.now() - timedelta(days=days_back)
        since = int(start_date.timestamp() * 1000)

        all_data = []

        last_ts = None
        same_count = 0

        while True:
            data = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=max_limit
            )

            if not data:
                break

            current_last_ts = data[-1][0]

            # 👉 проверяем только между итерациями
            if last_ts is not None:
                if current_last_ts == last_ts:
                    same_count += 1
                else:
                    same_count = 0

            if same_count >= 3:
                print("⚠️ Stuck (no new candles), stopping")
                break

            last_ts = current_last_ts

            all_data.extend(data)

            since = current_last_ts + 1

            print(f"Fetched {len(data)} | Total: {len(all_data)}")

            time.sleep(0.2)

        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df.sort_values('timestamp').reset_index(drop=True)

    def _timeframe_to_minutes(self, timeframe):
        tf_map = {'15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240, '1d': 1440, '1w': 10080}
        return tf_map.get(timeframe, 60)

    def fetch_recent_history(self, symbol, timeframe, since):
        max_limit = 1000
        all_data = []

        while True:
            data = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=max_limit
            )

            if not data:
                break

            all_data.extend(data)

            new_since = data[-1][0] + 1

            print(f"Fetched {len(data)} new candles")

            # если нет прогресса — стоп
            if new_since <= since:
                print("⚠️ No progress, stopping")
                break

            since = new_since

            if len(data) < max_limit:
                break

            time.sleep(0.2)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df.sort_values('timestamp').reset_index(drop=True)

# Пример использования
if __name__ == "__main__":
    client = BybitClient(testnet=False)

    # Скачиваем ВСЮ доступную историю для BTCUSDT на 1h
    df = client.fetch_all_history('BTC/USDT', '1h')
    print(f"\n✅ Получено {len(df)} свечей")
    print(df.head())
    print(df.tail())
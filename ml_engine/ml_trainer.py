import numpy as np
from ml_engine.ml_model import MLModel


class MLTrainer:
    def __init__(self, db, symbols, timeframes):
        self.db = db
        self.symbols = symbols
        self.timeframes = timeframes
        self.model = MLModel()
    def build_mtf_state(self, symbol, timestamp):
        state = {}

        for tf in self.timeframes:
            df = self.db.load_ohlcv(symbol, tf)

            if df.empty:
                continue

            # Р±РµСЂС‘Рј РІСЃРµ СЃРІРµС‡Рё Р”Рћ С‚РµРєСѓС‰РµРіРѕ РјРѕРјРµРЅС‚Р°
            df_filtered = df[df['timestamp'] < timestamp]

            if df_filtered.empty:
                continue

            # РїРѕСЃР»РµРґРЅСЏСЏ РґРѕСЃС‚СѓРїРЅР°СЏ СЃРІРµС‡Р°
            tf_row = df_filtered.iloc[-1]

            state[tf] = {
                'close': tf_row['close'],

                'rsi': tf_row['rsi'],
                'rsi_ema': tf_row['rsi_ema'],

                'macd': tf_row['macd'],
                'macd_signal': tf_row['macd_signal'],
                'macd_hist': tf_row['macd_hist'],

                'bb_mid': tf_row['bb_mid'],
                'bb_width': tf_row['bb_width'],

                'nw_mid': tf_row['nw_mid'],
                'nw_upper': tf_row['nw_upper'],
                'nw_lower': tf_row['nw_lower'],

                'atr': tf_row['atr'],
                'atr_exp': tf_row['atr_exp'],

                'structure': tf_row['structure'],
                'slope_cons': tf_row['slope_cons'],
                'near_edge': tf_row['near_edge'],

                'squeeze': tf_row['squeeze'],

                'score_short': tf_row['score_short'],
                'score_long': tf_row['score_long'],

                'bias_short': tf_row['bias_short'],
                'bias_long': tf_row['bias_long'],

                'composite': tf_row['composite'],
                'regime': tf_row['regime'],
            }

        return state
    def build_dataset(self):
        X = []
        y = []

        for symbol in self.symbols:
            tf_main = list(self.timeframes.keys())[0]

            df = self.db.load_ohlcv(symbol, tf_main)

            if df.empty or len(df) < 200:
                continue

            for i in range(len(df) - 20):
                row = df.iloc[i]

                timestamp = row['timestamp']
                state = self.build_mtf_state(symbol, timestamp)
                if len(state) < len(self.timeframes):
                    continue

                future_price = df.iloc[i + 10]['close']
                future_price_far = df.iloc[i + 20]['close']

                current_price = row['close']



                # С„РёР»СЊС‚СЂСѓРµРј СЃР»Р°Р±С‹Рµ РґРІРёР¶РµРЅРёСЏ
                move = abs((future_price_far - current_price) / current_price)

                if move < 0.003:  # <0.3% вЂ” С€СѓРј
                    continue
                current_price = row['close']

                target = 1 if future_price > current_price else 0

                features = self.model.build_features(state)

                X.append(features)
                y.append(target)

        return np.array(X), np.array(y)

    def train(self):
        X, y = self.build_dataset()

        if len(X) < 50:
            print("[WARN] Not enough data for ML")
            return

        self.model.train(X, y)
        self.model.save("ml_model.pkl")

        print(f"[OK] ML trained on {len(X)} samples")
        from collections import Counter

        print("Class balance:", Counter(y))
        return self.model

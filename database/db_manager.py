import sqlite3
import pandas as pd
from datetime import datetime
import os
import time

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path

        # СЃРѕР·РґР°С‘Рј Р‘Р” Рё РѕСЃРЅРѕРІРЅС‹Рµ С‚Р°Р±Р»РёС†С‹
        self._init_database()

        # РѕС‚РєСЂС‹РІР°РµРј СЃРѕРµРґРёРЅРµРЅРёРµ РґР»СЏ РЅР°СЃС‚СЂРѕРµРє Рё trades
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")

            conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                type TEXT,
                entry REAL,
                exit REAL,
                pnl REAL,
                pnl_pct REAL,
                open_time INTEGER,
                close_time INTEGER,
                timeframe TEXT
            )
            """)

    def _format_symbol(self, symbol):
        return symbol.replace("/", "_").replace(":", "_")
    def get_stats(self, days):
        with sqlite3.connect(self.db_path) as conn:
            since = int(time.time()) - days * 86400

            df = pd.read_sql_query(f"""
            SELECT * FROM trades
            WHERE close_time >= {since}
            """, conn)

        if df.empty:
            return None

        total = len(df)
        wins = len(df[df['pnl'] > 0])
        winrate = wins / total * 100

        avg_pnl = df['pnl'].mean()
        total_pnl = df['pnl'].sum()

        return {
            'total': total,
            'winrate': winrate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl
        }
    def save_trade(self, trade_data):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            INSERT INTO trades (
                symbol, type, entry, exit,
                pnl, pnl_pct, open_time, close_time, timeframe
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, trade_data)
    def _init_database(self):
        """РРЅРёС†РёР°Р»РёР·РёСЂСѓРµС‚ Р‘Р” Рё СЃРѕР·РґР°РµС‚ СЃР»СѓР¶РµР±РЅС‹Рµ С‚Р°Р±Р»РёС†С‹"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # РўР°Р±Р»РёС†Р° РґР»СЏ С…СЂР°РЅРµРЅРёСЏ СЃРёРіРЅР°Р»РѕРІ
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timestamp INTEGER,
                    timeframe TEXT,
                    signal_type TEXT,
                    entry REAL,
                    tp REAL,
                    sl REAL,
                    composite REAL,
                    regime TEXT,
                    status TEXT,
                    closed_at INTEGER,
                    pnl REAL
                )
            ''')

            # РўР°Р±Р»РёС†Р° РґР»СЏ СЃС‚Р°С‚РёСЃС‚РёРєРё РїР°С‚С‚РµСЂРЅРѕРІ
            conn.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_hash TEXT UNIQUE,
                    regime TEXT,
                    composite_range TEXT,
                    bias_short TEXT,
                    bias_long TEXT,
                    probability REAL,
                    expected_move_pct REAL,
                    win_rate REAL,
                    samples INTEGER,
                    last_updated INTEGER
                )
            ''')

            # РўР°Р±Р»РёС†Р° РґР»СЏ Р»РѕРіРѕРІ РѕР±СѓС‡РµРЅРёСЏ
            conn.execute('''
                CREATE TABLE IF NOT EXISTS training_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    symbols_processed INTEGER,
                    patterns_found INTEGER,
                    duration_seconds REAL
                )
            ''')

    def create_timeframe_table(self, symbol, timeframe):
        """РЎРѕР·РґР°РµС‚ С‚Р°Р±Р»РёС†Сѓ РґР»СЏ РєРѕРЅРєСЂРµС‚РЅРѕРіРѕ С‚РѕРєРµРЅР° Рё С‚Р°Р№РјС„СЂРµР№РјР°"""
        symbol_clean = self._format_symbol(symbol)
        table_name = f"{symbol_clean}_{timeframe}"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    timestamp INTEGER PRIMARY KEY,
                    open REAL, 
                    high REAL, 
                    low REAL, 
                    close REAL, 
                    volume REAL,
                    rsi REAL, 
                    rsi_ema REAL,
                    macd REAL, 
                    macd_signal REAL, 
                    macd_hist REAL, 
                    impulse_growing INTEGER,
                    nw_mid REAL, 
                    nw_upper REAL, 
                    nw_lower REAL,
                    bb_mid REAL, 
                    bb_upper REAL, 
                    bb_lower REAL, 
                    bb_width REAL, 
                    squeeze REAL,
                    atr REAL, 
                    atr_exp REAL,
                    slope_cons REAL, 
                    structure REAL, 
                    near_edge REAL,
                    score_short REAL, 
                    score_long REAL,
                    bias_short TEXT, 
                    bias_long TEXT,
                    composite REAL, 
                    regime TEXT,
                    long_signal INTEGER, 
                    short_signal INTEGER
                )
            ''')
            print(f"[OK] Table {table_name} created/verified")

    def save_ohlcv(self, symbol, timeframe, df):
        """РЎРѕС…СЂР°РЅСЏРµС‚ OHLCV РґР°РЅРЅС‹Рµ РІ Р‘Р”"""
        symbol_clean = self._format_symbol(symbol)
        table_name = f"{symbol_clean}_{timeframe}"

        # РЎРЅР°С‡Р°Р»Р° СЃРѕР·РґР°РµРј С‚Р°Р±Р»РёС†Сѓ РµСЃР»Рё РµС‘ РЅРµС‚
        self.create_timeframe_table(symbol, timeframe)

        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            print(f"[OK] Saved {len(df)} rows to {table_name}")

    def load_ohlcv(self, symbol, timeframe):
        """Р—Р°РіСЂСѓР¶Р°РµС‚ OHLCV РґР°РЅРЅС‹Рµ РёР· Р‘Р”"""
        symbol_clean = self._format_symbol(symbol)
        table_name = f"{symbol_clean}_{timeframe}"

        try:
            with sqlite3.connect(self.db_path) as conn:
                # РџСЂРѕРІРµСЂСЏРµРј СЃСѓС‰РµСЃС‚РІСѓРµС‚ Р»Рё С‚Р°Р±Р»РёС†Р°
                cursor = conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                if cursor.fetchone() is None:
                    print(f"[WARN] Table {table_name} does not exist yet")
                    return pd.DataFrame()

                df = pd.read_sql_query(f'SELECT * FROM {table_name} ORDER BY timestamp ASC', conn)
                return df
        except Exception as e:
            print(f"Error loading {table_name}: {e}")
            return pd.DataFrame()

    def save_signal(self, signal_data):
        """РЎРѕС…СЂР°РЅСЏРµС‚ СЃРёРіРЅР°Р» РІ Р‘Р”"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO signals 
                (symbol, timestamp, timeframe, signal_type, entry, tp, sl, composite, regime, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', signal_data)
            return cursor.lastrowid

    def update_signal_status(self, signal_id, status, closed_at=None, pnl=None):
        """РћР±РЅРѕРІР»СЏРµС‚ СЃС‚Р°С‚СѓСЃ СЃРёРіРЅР°Р»Р°"""
        with sqlite3.connect(self.db_path) as conn:
            if closed_at and pnl is not None:
                conn.execute('''
                    UPDATE signals 
                    SET status=?, closed_at=?, pnl=?
                    WHERE id=?
                ''', (status, closed_at, pnl, signal_id))
            else:
                conn.execute('''
                    UPDATE signals SET status=? WHERE id=?
                ''', (status, signal_id))

    def save_pattern(self, pattern_hash, probability, expected_move, win_rate, samples, timestamp):
        """РЎРѕС…СЂР°РЅСЏРµС‚ РїР°С‚С‚РµСЂРЅ РІ Р‘Р”"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO patterns 
                (pattern_hash, probability, expected_move_pct, win_rate, samples, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (pattern_hash, probability, expected_move, win_rate, samples, timestamp))

    def load_pattern(self, pattern_hash, lookahead):
        """Р—Р°РіСЂСѓР¶Р°РµС‚ РїР°С‚С‚РµСЂРЅ РёР· Р‘Р”"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT probability, expected_move_pct FROM patterns 
                WHERE pattern_hash=? LIMIT 1
            ''', (pattern_hash,))
            row = cursor.fetchone()
            if row:
                return {
                    'probability': row[0],
                    'expected_move_pct': row[1]
                }
        return 0.5, 0.5

    def get_statistics(self, symbol, days):
        """РџРѕР»СѓС‡Р°РµС‚ СЃС‚Р°С‚РёСЃС‚РёРєСѓ РїРѕ СЃРёРіРЅР°Р»Р°Рј Р·Р° N РґРЅРµР№"""
        timestamp_limit = int((datetime.now().timestamp() - days * 86400) * 1000)

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT * FROM signals 
                WHERE symbol=? AND timestamp>=? AND status='closed'
                ORDER BY timestamp DESC
            ''', conn, params=(symbol, timestamp_limit))

            if df.empty:
                return None

            total_signals = len(df)
            winning = len(df[df['pnl'] > 0])
            win_rate = winning / total_signals if total_signals > 0 else 0
            avg_pnl = df['pnl'].mean()
            total_pnl = df['pnl'].sum()

            return {
                'total_signals': total_signals,
                'winning': winning,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_pnl': total_pnl,
                'best_trade': df['pnl'].max(),
                'worst_trade': df['pnl'].min()
            }

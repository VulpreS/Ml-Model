import asyncio
import sqlite3
import time
from datetime import datetime

import schedule


class WeeklyTrainer:
    def __init__(self, db_manager, pattern_learner, data_updater, symbols, timeframes):
        self.db = db_manager
        self.pattern_learner = pattern_learner
        self.data_updater = data_updater
        self.symbols = symbols
        self.timeframes = timeframes
        self.is_running = True
        self.pattern_analyzer = None

    def _log(self, message):
        print(f"[WeeklyTrainer] {message}")

    def start_scheduler(self):
        schedule.every().monday.at("00:00").do(self.run_training_sync)
        self._log("Scheduler started. Next training: Monday 00:00 UTC")

        while self.is_running:
            schedule.run_pending()
            time.sleep(60)

    def run_training_sync(self):
        asyncio.run(self.run_training())

    async def run_training(self):
        self._log(f"Training started at {datetime.now().isoformat(sep=' ', timespec='seconds')}")
        start_time = time.time()

        self._log("Step 1/3: updating recent data")
        await asyncio.to_thread(
            self.data_updater.update_recent_data,
            self.symbols,
            self.timeframes,
            168,
        )

        self._log("Step 2/3: regenerating signals")
        await self._regenerate_all_signals()

        self._log("Step 3/3: deep pattern learning")
        from ml_engine.pattern_analyzer import PatternAnalyzer

        analyzer = PatternAnalyzer(self.db.db_path)
        await asyncio.to_thread(
            analyzer.learn_from_database,
            self.symbols,
            self.timeframes,
            5,
        )
        self.pattern_analyzer = analyzer

        duration = time.time() - start_time
        self._save_training_log(len(self.symbols), duration)
        self._log(f"Training completed in {duration:.2f}s")

    async def _regenerate_all_signals(self):
        from pine_logic.signal_generator import SignalGenerator

        generator = SignalGenerator(mode="SCALP")

        for symbol in self.symbols:
            for tf_name, tf_minutes in self.timeframes.items():
                self._log(f"Regenerating signals for {symbol} {tf_name}")
                df = self.db.load_ohlcv(symbol, tf_name)

                if df.empty or len(df) < 100:
                    self._log(f"Skipping {symbol} {tf_name}: not enough data")
                    continue

                htf_name = generator.get_htf(tf_minutes)
                if htf_name == "D":
                    htf_name = "1d"
                elif htf_name == "240":
                    htf_name = "4h"
                elif htf_name == "120":
                    htf_name = "2h"
                elif htf_name == "60":
                    htf_name = "1h"

                htf_df = None
                if htf_name in self.timeframes:
                    htf_df = self.db.load_ohlcv(symbol, htf_name)

                df_with_signals = generator.generate_signals_for_timeframe(df, tf_minutes, htf_df)
                self.db.save_ohlcv(symbol, tf_name, df_with_signals)

    def _save_training_log(self, symbols_processed, duration):
        with sqlite3.connect(self.db.db_path) as conn:
            conn.execute(
                """
                INSERT INTO training_log (timestamp, symbols_processed, patterns_found, duration_seconds)
                VALUES (?, ?, ?, ?)
                """,
                (
                    int(datetime.now().timestamp() * 1000),
                    symbols_processed,
                    0,
                    duration,
                ),
            )

    def stop(self):
        self.is_running = False

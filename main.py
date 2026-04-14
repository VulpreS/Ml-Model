#!/usr/bin/env python3
"""
Quantum Signal Agent - Autonomous Multi-Timeframe Signal Generator
"""

import asyncio
import logging
import signal
import sys
import threading
import time
from pathlib import Path

from config import (
    BYBIT_API_KEY,
    BYBIT_API_SECRET,
    DB_PATH,
    ML_MODEL_PATH,
    SYMBOLS,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TIMEFRAMES,
)
from data_fetcher.data_updater import DataUpdater
from database.db_manager import DatabaseManager
from ml_engine.ml_trainer import MLTrainer
from ml_engine.pattern_analyzer import PatternAnalyzer
from ml_engine.pattern_learner import PatternLearner
from ml_engine.weekly_trainer import WeeklyTrainer
from pine_logic.signal_generator import SignalGenerator
from telegram_bot.bot import TelegramBot


def create_directories():
    directories = ["database", "logs", "data_fetcher", "pine_logic", "ml_engine", "telegram_bot", "utils"]
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        init_file = path / "__init__.py"
        if not init_file.exists():
            init_file.touch()


def configure_console_output():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


create_directories()
configure_console_output()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/signal_agent.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class QuantumSignalAgent:
    def __init__(self):
        logger.info("Initializing Quantum Signal Agent...")

        self.db = DatabaseManager(DB_PATH)
        self.symbols = SYMBOLS
        self.timeframes = TIMEFRAMES
        self.data_updater = DataUpdater(self.db, BYBIT_API_KEY, BYBIT_API_SECRET)
        self.signal_generator = SignalGenerator(mode="SCALP")
        self.pattern_learner = PatternLearner(self.db)
        self.pattern_analyzer = PatternAnalyzer(DB_PATH)
        self.weekly_trainer = WeeklyTrainer(
            self.db,
            self.pattern_learner,
            self.data_updater,
            self.symbols,
            self.timeframes,
        )
        self.ml_trainer = MLTrainer(self.db, self.symbols, self.timeframes)
        self.ml_model = self.ml_trainer.model
        self.telegram_bot = None
        self.is_running = False
        self.last_bar_check = {}
        self.active_signals = {}

        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            self.telegram_bot = TelegramBot(
                TELEGRAM_BOT_TOKEN,
                TELEGRAM_CHAT_ID,
                self.signal_generator,
                self.db,
                self.pattern_learner,
                weekly_trainer=self.weekly_trainer,
            )

        self.ml_model.load(ML_MODEL_PATH)
        logger.info("Agent initialized successfully")

    async def initialize(self):
        logger.info("Starting initial data loading...")

        data_exists = False
        for symbol in self.symbols:
            for timeframe in self.timeframes.keys():
                df = self.db.load_ohlcv(symbol, timeframe)
                if not df.empty:
                    data_exists = True
                    break
            if data_exists:
                break

        if not data_exists:
            logger.info("No data found. Loading historical data from Bybit...")
            await asyncio.to_thread(
                self.data_updater.update_recent_data,
                self.symbols,
                self.timeframes,
            )
        else:
            logger.info("Existing data found. Updating recent data...")
            await asyncio.to_thread(
                self.data_updater.update_recent_data,
                self.symbols,
                self.timeframes,
                168,
            )

        logger.info("Initial data loading complete")

        if not self.ml_model.is_trained:
            logger.info("Training ML model...")
            self.ml_model = self.ml_trainer.train()

        logger.info("Starting initial pattern learning...")
        await self.weekly_trainer.run_training()
        logger.info("Initialization complete")

    async def check_for_new_bars(self):
        current_time = time.time()

        for symbol in self.symbols:
            for tf_name, tf_minutes in self.timeframes.items():
                last_check_key = f"{symbol}_{tf_name}"
                last_check = self.last_bar_check.get(last_check_key, 0)

                if current_time - last_check < 60:
                    continue

                df = self.db.load_ohlcv(symbol, tf_name)
                if df.empty or len(df) < 2:
                    self.last_bar_check[last_check_key] = current_time
                    continue

                last_candle = df.iloc[-1]
                prev_candle = df.iloc[-2]
                last_timestamp = self._normalize_timestamp(last_candle["timestamp"])
                current_bar_time = self._get_bar_timestamp(current_time, tf_minutes)

                if last_timestamp < current_bar_time:
                    if prev_candle.get("long_signal", 0) == 1 or prev_candle.get("short_signal", 0) == 1:
                        signal_type = "LONG" if prev_candle.get("long_signal", 0) == 1 else "SHORT"

                        if symbol not in self.active_signals:
                            logger.info(f"New {signal_type} signal detected for {symbol} on {tf_name}")
                            signal_data = await self._generate_signal_data(symbol, tf_name, signal_type, prev_candle)

                            if signal_data is None:
                                logger.warning(
                                    "Skipping signal for %s %s because MTF state is incomplete",
                                    symbol,
                                    tf_name,
                                )
                            else:
                                if self.telegram_bot:
                                    await self.telegram_bot.send_signal(signal_data)

                                signal_id = self.db.save_signal(
                                    (
                                        symbol,
                                        self._normalize_timestamp(prev_candle["timestamp"]),
                                        tf_name,
                                        signal_data["signal_type"],
                                        float(prev_candle["close"]),
                                        signal_data["tp"],
                                        signal_data["sl"],
                                        float(prev_candle.get("composite", 0)),
                                        prev_candle.get("regime", "NEUTRAL"),
                                        "active",
                                    )
                                )

                                self.active_signals[symbol] = {
                                    "id": signal_id,
                                    "type": signal_data["signal_type"],
                                    "entry": float(prev_candle["close"]),
                                    "tp": signal_data["tp"],
                                    "sl": signal_data["sl"],
                                    "timestamp": self._normalize_timestamp(prev_candle["timestamp"]),
                                    "timeframe": tf_name,
                                }

                self.last_bar_check[last_check_key] = current_time

        await self._check_active_signals()

    async def _generate_signal_data(self, symbol, tf_name, signal_type, candle):
        timestamp = self._normalize_timestamp(candle["timestamp"])
        current_state = self.ml_trainer.build_mtf_state(symbol, timestamp)

        if len(current_state) < len(self.timeframes):
            return None

        timeframes_data = {}
        for tf, state in current_state.items():
            tf_signal = "NEUTRAL"
            if state.get("score_long", 0) > state.get("score_short", 0):
                tf_signal = "LONG"
            elif state.get("score_short", 0) > state.get("score_long", 0):
                tf_signal = "SHORT"

            timeframes_data[tf] = {
                "signal": tf_signal,
                "composite": state.get("composite", 0),
                "regime": state.get("regime", "NEUTRAL"),
                "bias_short": state.get("bias_short", "NEUTRAL"),
                "bias_long": state.get("bias_long", "NEUTRAL"),
            }

        prediction = self.ml_model.predict(current_state) if self.ml_model else {
            "direction": signal_type,
            "probability": 0.5,
            "expected_move": 1.0,
            "tp_pct": 1.5,
            "sl_pct": 1.0,
        }

        final_direction = prediction["direction"] if prediction["direction"] != "NEUTRAL" else signal_type
        entry = float(candle["close"])

        if final_direction == "LONG":
            tp = entry * (1 + prediction["tp_pct"] / 100)
            sl = entry * (1 - prediction["sl_pct"] / 100)
            tp_pct = prediction["tp_pct"]
            sl_pct = -prediction["sl_pct"]
        else:
            tp = entry * (1 - prediction["tp_pct"] / 100)
            sl = entry * (1 + prediction["sl_pct"] / 100)
            tp_pct = -prediction["tp_pct"]
            sl_pct = prediction["sl_pct"]

        active_biases = [
            state["bias_short"]
            for state in current_state.values()
            if state.get("bias_short") != "NEUTRAL"
        ]
        conflict_detected = len(set(active_biases)) > 1
        conflict_reason = None
        if conflict_detected:
            conflict_reason = f"Bias mismatch: { {tf: state.get('bias_short') for tf, state in current_state.items()} }"

        return {
            "symbol": symbol,
            "signal_type": final_direction,
            "entry": entry,
            "tp": float(tp),
            "sl": float(sl),
            "tp_pct": float(tp_pct),
            "sl_pct": float(sl_pct),
            "probability": float(prediction["probability"]),
            "expected_move": float(prediction["expected_move"]),
            "samples": prediction.get("samples", 0),
            "win_rate": prediction.get("win_rate", prediction["probability"]),
            "composite": float(candle.get("composite", 0)),
            "regime": candle.get("regime", "NEUTRAL"),
            "bias_short": candle.get("bias_short", "NEUTRAL"),
            "bias_long": candle.get("bias_long", "NEUTRAL"),
            "score_short": float(candle.get("score_short", 0)),
            "score_long": float(candle.get("score_long", 0)),
            "timeframes": timeframes_data,
            "conflict": conflict_detected,
            "conflict_reason": conflict_reason,
            "timestamp": timestamp,
            "timeframe": tf_name,
            "close_price": entry,
        }

    async def _check_active_signals(self):
        for symbol in list(self.active_signals.keys()):
            active_signal = self.active_signals[symbol]

            try:
                df = self.db.load_ohlcv(symbol, active_signal["timeframe"])
                if df.empty:
                    continue

                last_candle = df.iloc[-1]
                current_price = float(last_candle["close"])
                should_close = False
                close_reason = ""

                if active_signal["type"] == "LONG":
                    if current_price >= active_signal["tp"]:
                        should_close = True
                        close_reason = "TP hit"
                    elif current_price <= active_signal["sl"]:
                        should_close = True
                        close_reason = "SL hit"
                else:
                    if current_price <= active_signal["tp"]:
                        should_close = True
                        close_reason = "TP hit"
                    elif current_price >= active_signal["sl"]:
                        should_close = True
                        close_reason = "SL hit"

                current_signal = "NEUTRAL"
                if last_candle.get("long_signal", 0) == 1:
                    current_signal = "LONG"
                elif last_candle.get("short_signal", 0) == 1:
                    current_signal = "SHORT"

                if current_signal != "NEUTRAL" and current_signal != active_signal["type"]:
                    should_close = True
                    close_reason = f"Signal reversed ({current_signal})"

                conflict_count = 0
                for tf_name in self.timeframes.keys():
                    df_tf = self.db.load_ohlcv(symbol, tf_name)
                    if df_tf.empty:
                        continue
                    if df_tf.iloc[-1].get("regime") == "CONFLICT":
                        conflict_count += 1

                if conflict_count >= 3:
                    should_close = True
                    close_reason = f"MTF conflict ({conflict_count})"

                if should_close:
                    if active_signal["type"] == "LONG":
                        pnl_pct = (current_price - active_signal["entry"]) / active_signal["entry"]
                    else:
                        pnl_pct = (active_signal["entry"] - current_price) / active_signal["entry"]

                    pnl = pnl_pct * 100
                    self.db.save_trade(
                        (
                            symbol,
                            active_signal["type"],
                            active_signal["entry"],
                            current_price,
                            pnl,
                            pnl_pct,
                            active_signal["timestamp"],
                            int(time.time()),
                            active_signal["timeframe"],
                        )
                    )

                    logger.info(
                        "%s CLOSED | %s | %s | PnL: %.2f%%",
                        symbol,
                        active_signal["type"],
                        close_reason,
                        pnl,
                    )

                    if self.telegram_bot:
                        await self.telegram_bot.send_message(
                            (
                                f"Trade closed\n\n"
                                f"Symbol: {symbol}\n"
                                f"Type: {active_signal['type']}\n"
                                f"Reason: {close_reason}\n"
                                f"Entry: {active_signal['entry']:.2f}\n"
                                f"Exit: {current_price:.2f}\n"
                                f"PnL: {pnl:.2f}%"
                            )
                        )

                    del self.active_signals[symbol]

            except Exception as exc:
                logger.exception("Error checking signal %s: %s", symbol, exc)

    def _normalize_timestamp(self, value):
        if hasattr(value, "timestamp"):
            return int(value.timestamp() * 1000)
        return int(value)

    def _get_bar_timestamp(self, current_time, tf_minutes):
        bar_number = int(current_time // (tf_minutes * 60))
        return bar_number * (tf_minutes * 60) * 1000

    async def run(self):
        logger.info("Starting Quantum Signal Agent...")
        await self.initialize()

        if self.telegram_bot:
            asyncio.create_task(self.telegram_bot.start())
            logger.info("Telegram bot started")

        trainer_thread = threading.Thread(target=self.weekly_trainer.start_scheduler, daemon=True)
        trainer_thread.start()
        logger.info("Weekly trainer scheduler started")

        self.is_running = True
        while self.is_running:
            try:
                await self.check_for_new_bars()
                await asyncio.sleep(60)
            except Exception as exc:
                logger.exception("Error in main loop: %s", exc)
                await asyncio.sleep(5)

    def stop(self):
        logger.info("Stopping Quantum Signal Agent...")
        self.is_running = False
        self.weekly_trainer.stop()


def signal_handler(signum, frame):
    logger.info("Received shutdown signal")
    if "agent" in globals() and hasattr(agent, "stop"):
        agent.stop()
    sys.exit(0)


if __name__ == "__main__":
    agent = QuantumSignalAgent()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
    except Exception as exc:
        logger.exception("Agent crashed: %s", exc)
        sys.exit(1)

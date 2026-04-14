#!/usr/bin/env python3
"""
Quantum Signal Agent - Autonomous Multi-Timeframe Signal Generator
Based on Market Regime & Signal Scoring Model
"""

import os
import sys
import time
import asyncio
import signal
import logging
import threading
from datetime import datetime
from pathlib import Path


# ===== РЎРѕР·РґР°РµРј РїР°РїРєРё Р”Рћ Р»РѕРіРёСЂРѕРІР°РЅРёСЏ =====
def create_directories():
    directories = ['database', 'logs', 'data_fetcher', 'pine_logic', 'ml_engine', 'telegram_bot', 'utils']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        init_file = Path(directory) / '__init__.py'
        if not init_file.exists():
            init_file.touch()


create_directories()
def configure_console_output():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


configure_console_output()

# ===== Р›РѕРіРёСЂРѕРІР°РЅРёРµ =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/signal_agent.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ===== РРјРїРѕСЂС‚С‹ =====
from config import *
from database.db_manager import DatabaseManager
from data_fetcher.data_updater import DataUpdater
from pine_logic.signal_generator import SignalGenerator
from ml_engine.pattern_learner import PatternLearner
from ml_engine.weekly_trainer import WeeklyTrainer
from ml_engine.pattern_analyzer import PatternAnalyzer
from telegram_bot.bot import TelegramBot
from ml_engine.ml_trainer import MLTrainer

class QuantumSignalAgent:
    def __init__(self):
        logger.info("Initializing Quantum Signal Agent...")

        self.db = DatabaseManager(DB_PATH)
        self.symbols = SYMBOLS
        self.timeframes = TIMEFRAMES
        self.data_updater = DataUpdater(self.db, BYBIT_API_KEY, BYBIT_API_SECRET)
        self.signal_generator = SignalGenerator(mode='SCALP')
        self.pattern_learner = PatternLearner(self.db)
        self.pattern_analyzer = PatternAnalyzer(DB_PATH)
        self.weekly_trainer = WeeklyTrainer(
            self.db, self.pattern_learner, self.data_updater,
            SYMBOLS, TIMEFRAMES
        )
        self.ml_trainer = MLTrainer(self.db, SYMBOLS, TIMEFRAMES)
        self.ml_model = None
        self.telegram_bot = None
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            self.telegram_bot = TelegramBot(
                TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                self.signal_generator, self.db, self.pattern_learner
            )
        self.ml_model = self.ml_trainer.train()
        self.is_running = False
        self.last_bar_check = {}
        self.active_signals = {}

        logger.info("Agent initialized successfully")

    async def initialize(self):
        logger.info("Starting initial data loading...")

        data_exists = False
        for symbol in SYMBOLS:
            for tf in TIMEFRAMES.keys():
                df = self.db.load_ohlcv(symbol, tf)
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
                self.timeframes
            )
        else:
            logger.info("Existing data found. Updating recent data...")
            await asyncio.to_thread(
                self.data_updater.update_recent_data,
                SYMBOLS, TIMEFRAMES, 168
            )

        logger.info("Initial data loading complete")

        logger.info("Starting initial pattern learning...")
        await self.weekly_trainer.run_training()

        logger.info("Initialization complete!")

    async def check_for_new_bars(self):
        current_time = time.time()

        for symbol in SYMBOLS:
            for tf_name, tf_minutes in TIMEFRAMES.items():
                last_check_key = f"{symbol}_{tf_name}"
                last_check = self.last_bar_check.get(last_check_key, 0)

                if current_time - last_check < 60:
                    continue

                df = self.db.load_ohlcv(symbol, tf_name)
                if df.empty or len(df) < 2:
                    continue

                last_candle = df.iloc[-1]
                prev_candle = df.iloc[-2]

                last_timestamp = last_candle['timestamp']
                current_bar_time = self._get_bar_timestamp(current_time, tf_minutes)

                if last_timestamp < current_bar_time:
                    if prev_candle.get('long_signal', 0) == 1 or prev_candle.get('short_signal', 0) == 1:
                        signal_type = 'LONG' if prev_candle['long_signal'] == 1 else 'SHORT'

                        if symbol not in self.active_signals:
                            logger.info(f"New {signal_type} signal detected for {symbol} on {tf_name}")

                            signal_data = await self._generate_signal_data(
                                symbol, tf_name, signal_type, prev_candle
                            )

                            if self.telegram_bot:
                                await self.telegram_bot.send_signal(signal_data)

                            signal_id = self.db.save_signal((
                                symbol, int(prev_candle['timestamp']), tf_name,
                                signal_type, prev_candle['close'],
                                signal_data['tp'], signal_data['sl'],
                                prev_candle['composite'], prev_candle['regime'],
                                'active'
                            ))

                            self.active_signals[symbol] = {
                                'id': signal_id,
                                'type': signal_type,
                                'entry': prev_candle['close'],
                                'tp': signal_data['tp'],
                                'sl': signal_data['sl'],
                                'timestamp': prev_candle['timestamp'],
                                'timeframe': tf_name
                            }

                self.last_bar_check[last_check_key] = current_time

        await self._check_active_signals()

    async def _generate_signal_data(self, symbol, tf_name, signal_type, candle):
        """Р“РµРЅРµСЂРёСЂСѓРµС‚ РґР°РЅРЅС‹Рµ СЃРёРіРЅР°Р»Р° СЃ РёСЃРїРѕР»СЊР·РѕРІР°РЅРёРµРј ML"""

        # =========================
        # рџ§  РџР РђР’РР›Р¬РќР«Р™ MTF STATE
        # =========================
        timestamp = candle['timestamp']

        current_state = self.ml_trainer.build_mtf_state(symbol, timestamp)

        # РµСЃР»Рё РЅРµС‚ РІСЃРµС… С‚Р°Р№РјС„СЂРµР№РјРѕРІ вЂ” РїСЂРѕРїСѓСЃРєР°РµРј
        if len(current_state) < len(TIMEFRAMES):
            return None

        # =========================
        # рџ“Љ Р”РђРќРќР«Р• Р”Р›РЇ РћРўРћР‘Р РђР–Р•РќРРЇ (Telegram)
        # =========================
        timeframes_data = {}

        for tf, s in current_state.items():
            signal_display = (
                'LONG' if s.get('long_signal', 0) == 1 else
                'SHORT' if s.get('short_signal', 0) == 1 else
                'NEUTRAL'
            )

            timeframes_data[tf] = {
                'signal': signal_display,
                'composite': s.get('composite', 0),
                'regime': s.get('regime', 'NEUTRAL'),
                'bias_short': s.get('bias_short', 'NEUTRAL'),
                'bias_long': s.get('bias_long', 'NEUTRAL')
            }

        # =========================
        # рџ¤– ML РџР Р•Р”РЎРљРђР—РђРќРР•
        # =========================
        if self.ml_model:
            prediction = self.ml_model.predict(current_state)
        else:
            prediction = {
                'direction': signal_type,
                'probability': 0.5,
                'expected_move': 1.0,
                'tp_pct': 1.5,
                'sl_pct': 1.0
            }

        entry = candle['close']

        # =========================
        # рџЋЇ TP / SL
        # =========================
        if prediction['direction'] == 'LONG':
            tp = entry * (1 + prediction['tp_pct'] / 100)
            sl = entry * (1 - prediction['sl_pct'] / 100)
            tp_pct = prediction['tp_pct']
            sl_pct = -prediction['sl_pct']
        else:
            tp = entry * (1 - prediction['tp_pct'] / 100)
            sl = entry * (1 + prediction['sl_pct'] / 100)
            tp_pct = -prediction['tp_pct']
            sl_pct = prediction['sl_pct']

        # =========================
        # вљ пёЏ РљРћРќР¤Р›РРљРўР«
        # =========================
        conflict_detected = False
        conflict_reason = None

        biases = [
            s['bias_short']
            for s in current_state.values()
            if s.get('bias_short') != 'NEUTRAL'
        ]

        if len(set(biases)) > 1:
            conflict_detected = True
            conflict_reason = f"Bias mismatch: { {tf: s.get('bias_short') for tf, s in current_state.items()} }"

        # =========================
        # рџ“¦ RETURN
        # =========================
        return {
            'symbol': symbol,
            'signal_type': prediction['direction'],
            'entry': entry,
            'tp': tp,
            'sl': sl,
            'tp_pct': tp_pct,
            'sl_pct': sl_pct,
            'probability': prediction['probability'],
            'expected_move': prediction['expected_move'],
            'samples': prediction.get('samples', 0),
            'win_rate': prediction.get('win_rate', prediction['probability']),
            'composite': candle.get('composite', 0),
            'regime': candle.get('regime', 'NEUTRAL'),
            'bias_short': candle.get('bias_short', 'NEUTRAL'),
            'bias_long': candle.get('bias_long', 'NEUTRAL'),
            'score_short': candle.get('score_short', 0),
            'score_long': candle.get('score_long', 0),
            'timeframes': timeframes_data,
            'conflict': conflict_detected,
            'conflict_reason': conflict_reason,
            'timestamp': candle.get('timestamp', 0),
            'timeframe': tf_name,
            'close_price': entry
        }
    async def _check_active_signals(self):
        for symbol in list(self.active_signals.keys()):
            signal = self.active_signals[symbol]

            try:
                df = self.db.load_ohlcv(symbol, signal['timeframe'])
                if df.empty:
                    continue

                last_candle = df.iloc[-1]
                current_price = last_candle['close']

                should_close = False
                close_reason = ""

                # =========================
                # рџЋЇ TP / SL
                # =========================
                if signal['type'] == 'LONG':
                    if current_price >= signal['tp']:
                        should_close = True
                        close_reason = "TP hit"
                    elif current_price <= signal['sl']:
                        should_close = True
                        close_reason = "SL hit"
                else:
                    if current_price <= signal['tp']:
                        should_close = True
                        close_reason = "TP hit"
                    elif current_price >= signal['sl']:
                        should_close = True
                        close_reason = "SL hit"

                # =========================
                # рџ”„ РџСЂРѕРІРµСЂРєР° СЂРµРІРµСЂСЃР° СЃРёРіРЅР°Р»Р°
                # =========================
                current_signal = 'LONG' if last_candle.get('long_signal', 0) == 1 else \
                    'SHORT' if last_candle.get('short_signal', 0) == 1 else 'NEUTRAL'

                if current_signal != 'NEUTRAL' and current_signal != signal['type']:
                    should_close = True
                    close_reason = f"Signal reversed ({current_signal})"

                # =========================
                # вљ пёЏ РџСЂРѕРІРµСЂРєР° РєРѕРЅС„Р»РёРєС‚РѕРІ MTF
                # =========================
                conflict_count = 0

                for tf_name in TIMEFRAMES.keys():
                    df_tf = self.db.load_ohlcv(symbol, tf_name)
                    if df_tf.empty:
                        continue

                    tf_candle = df_tf.iloc[-1]

                    if tf_candle.get('regime') == 'CONFLICT':
                        conflict_count += 1

                if conflict_count >= 3:
                    should_close = True
                    close_reason = f"MTF conflict ({conflict_count})"

                # =========================
                # рџ’° Р—Р°РєСЂС‹С‚РёРµ СЃРґРµР»РєРё
                # =========================
                if should_close:

                    # PnL СЂР°СЃС‡РµС‚
                    if signal['type'] == 'LONG':
                        pnl_pct = (current_price - signal['entry']) / signal['entry']
                    else:
                        pnl_pct = (signal['entry'] - current_price) / signal['entry']

                    pnl = pnl_pct * 100

                    # СЃРѕС…СЂР°РЅСЏРµРј С‚СЂРµР№Рґ
                    self.db.save_trade((
                        symbol,
                        signal['type'],
                        signal['entry'],
                        current_price,
                        pnl,
                        pnl_pct,
                        signal['timestamp'],
                        int(time.time()),
                        signal['timeframe']
                    ))

                    # Р»РѕРі
                    logger.info(
                        f"{symbol} CLOSED | {signal['type']} | {close_reason} | "
                        f"PnL: {pnl:.2f}%"
                    )

                    # telegram СѓРІРµРґРѕРјР»РµРЅРёРµ
                    if self.telegram_bot:
                        await self.telegram_bot.send_message(
                            f"""
    рџ“‰ РЎРґРµР»РєР° Р·Р°РєСЂС‹С‚Р°

    РЎРёРјРІРѕР»: {symbol}
    РўРёРї: {signal['type']}
    РџСЂРёС‡РёРЅР°: {close_reason}

    Entry: {signal['entry']:.2f}
    Exit: {current_price:.2f}

    PnL: {pnl:.2f}%
    """
                        )

                    # СѓРґР°Р»СЏРµРј РёР· Р°РєС‚РёРІРЅС‹С…
                    del self.active_signals[symbol]

            except Exception as e:
                logger.exception(f"Error checking signal {symbol}: {e}")

    def _get_bar_timestamp(self, current_time, tf_minutes):
        bar_number = int(current_time // (tf_minutes * 60))
        return bar_number * (tf_minutes * 60)

    async def run(self):
        logger.info("Starting Quantum Signal Agent...")

        await self.initialize()

        if self.telegram_bot:
            asyncio.create_task(self.telegram_bot.start())
            logger.info("Telegram bot started")

        trainer_thread = threading.Thread(target=self.weekly_trainer.start_scheduler)
        trainer_thread.daemon = True
        trainer_thread.start()
        logger.info("Weekly trainer scheduler started")

        self.is_running = True

        while self.is_running:
            try:
                await self.check_for_new_bars()
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)

    def stop(self):
        logger.info("Stopping Quantum Signal Agent...")
        self.is_running = False
        self.weekly_trainer.stop()


def signal_handler(signum, frame):
    logger.info("Received shutdown signal")
    if hasattr(agent, 'stop'):
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
    except Exception as e:
        logger.error(f"Agent crashed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        logger.exception(e)

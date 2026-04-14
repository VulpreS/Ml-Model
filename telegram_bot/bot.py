import asyncio

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes
except ImportError:
    Update = None
    Application = None
    CommandHandler = None
    ContextTypes = None


class TelegramBot:
    def __init__(
        self,
        token,
        chat_id,
        signal_generator,
        db_manager,
        pattern_learner,
        weekly_trainer=None,
    ):
        self.token = token
        self.chat_id = chat_id
        self.signal_generator = signal_generator
        self.db = db_manager
        self.pattern_learner = pattern_learner
        self.weekly_trainer = weekly_trainer
        self.application = None

    async def start(self):
        if Application is None:
            print("[TelegramBot] python-telegram-bot is not installed, bot disabled")
            return

        self.application = Application.builder().token(self.token).build()
        self.application.add_handler(CommandHandler("signal", self.signal_command))
        self.application.add_handler(CommandHandler("stat", self.stat_command))
        self.application.add_handler(CommandHandler("learn", self.learn_command))
        await self.application.run_polling()

    async def signal_command(self, update, context):
        if len(context.args) == 0:
            await update.message.reply_text("Usage: /signal <symbol>\nExample: /signal BTC")
            return

        symbol = self._normalize_symbol(context.args[0].upper())
        signal_data = self._get_current_signal(symbol)

        if signal_data:
            await update.message.reply_text(
                self._format_signal_message(signal_data),
                parse_mode="HTML",
            )
        else:
            await update.message.reply_text(f"No signal available for {symbol}")

    async def stat_command(self, update, context):
        try:
            arg = context.args[0] if context.args else "7d"
            days = int(arg.replace("d", ""))
            stats = self.db.get_stats(days)

            if not stats:
                await update.message.reply_text("No trade data yet")
                return

            text = (
                f"Stats for {days} days\n\n"
                f"Trades: {stats['total']}\n"
                f"Winrate: {stats['winrate']:.2f}%\n"
                f"Avg PnL: {stats['avg_pnl']:.2f}%\n"
                f"Total PnL: {stats['total_pnl']:.2f}%"
            )
            await update.message.reply_text(text)
        except Exception as exc:
            await update.message.reply_text(f"Error: {exc}")

    async def learn_command(self, update, context):
        if self.weekly_trainer is None:
            await update.message.reply_text("Weekly trainer is not connected in this run")
            return

        await update.message.reply_text("Starting manual learning...")
        asyncio.create_task(self._run_learning())
        await update.message.reply_text("Learning started in background")

    def _normalize_symbol(self, symbol):
        if "/" not in symbol:
            return f"{symbol}/USDT:USDT"
        if ":USDT" not in symbol:
            return f"{symbol}:USDT"
        return symbol

    def _format_signal_message(self, signal):
        symbol = signal["symbol"].replace("/USDT:USDT", "").replace("/USDT", "")
        signal_type = signal["signal_type"].upper()
        emoji = "LONG" if signal_type == "LONG" else "SHORT" if signal_type == "SHORT" else "NEUTRAL"

        message = (
            f"<b>{emoji} {symbol}</b>\n"
            f"Regime: {signal['regime']} | Composite: {signal['composite']:.2f}\n"
            f"Bias: {signal['bias_short']} / {signal['bias_long']}\n\n"
            f"Entry: {signal['entry']:.2f}\n"
            f"TP: {signal['tp']:.2f} ({signal['tp_pct']:.2f}%)\n"
            f"SL: {signal['sl']:.2f} ({signal['sl_pct']:.2f}%)\n"
            f"Probability: {signal['probability'] * 100:.0f}%\n"
            f"Expected move: {signal['expected_move']:.2f}%\n\n"
            "Signals by TF:\n"
        )

        for tf, tf_signal in signal["timeframes"].items():
            message += f"{tf}: {tf_signal['signal']}\n"

        if signal.get("conflict"):
            message += f"\nCONFLICT: {signal['conflict_reason']}\n"

        return message

    def _get_current_signal(self, symbol):
        df = self.db.load_ohlcv(symbol, "15m")
        if df.empty or len(df) < 100:
            return None

        enriched = self.signal_generator.generate_signals_for_timeframe(df, 15)
        if enriched.empty:
            return None

        row = enriched.iloc[-1]
        if row.get("long_signal", 0) == 1:
            signal_type = "LONG"
        elif row.get("short_signal", 0) == 1:
            signal_type = "SHORT"
        else:
            signal_type = "NEUTRAL"

        entry = float(row["close"])
        tp_pct = 1.5
        sl_pct = 1.0

        if signal_type == "LONG":
            tp = entry * (1 + tp_pct / 100)
            sl = entry * (1 - sl_pct / 100)
            signed_tp_pct = tp_pct
            signed_sl_pct = -sl_pct
        elif signal_type == "SHORT":
            tp = entry * (1 - tp_pct / 100)
            sl = entry * (1 + sl_pct / 100)
            signed_tp_pct = -tp_pct
            signed_sl_pct = sl_pct
        else:
            tp = entry
            sl = entry
            signed_tp_pct = 0.0
            signed_sl_pct = 0.0

        signal = {
            "symbol": symbol,
            "signal_type": signal_type,
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "tp_pct": signed_tp_pct,
            "sl_pct": signed_sl_pct,
            "probability": 0.5,
            "expected_move": 0.5,
            "composite": float(row.get("composite", 0)),
            "regime": row.get("regime", "NEUTRAL"),
            "bias_short": row.get("bias_short", "NEUTRAL"),
            "bias_long": row.get("bias_long", "NEUTRAL"),
            "timeframes": {
                "15m": {
                    "signal": signal_type,
                    "composite": float(row.get("composite", 0)),
                    "regime": row.get("regime", "NEUTRAL"),
                    "bias_short": row.get("bias_short", "NEUTRAL"),
                    "bias_long": row.get("bias_long", "NEUTRAL"),
                }
            },
            "conflict": False,
            "conflict_reason": None,
        }

        if self.pattern_learner:
            prob, move = self.pattern_learner.get_prediction(
                signal["regime"],
                signal["composite"],
                signal["bias_short"],
                signal["bias_long"],
            )
            signal["probability"] = prob
            signal["expected_move"] = move

        return signal

    async def send_signal(self, signal_data):
        if not self.application:
            return

        try:
            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=self._format_signal_message(signal_data),
                parse_mode="HTML",
            )
        except Exception as exc:
            print(f"Error sending Telegram signal: {exc}")

    async def send_message(self, text):
        if not self.application:
            return

        try:
            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=text,
            )
        except Exception as exc:
            print(f"Error sending Telegram message: {exc}")

    async def _run_learning(self):
        if self.weekly_trainer is None:
            return
        await self.weekly_trainer.run_training()

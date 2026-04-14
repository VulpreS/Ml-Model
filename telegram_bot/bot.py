from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes
import asyncio
from datetime import datetime, timedelta
import pandas as pd


class TelegramBot:
    def __init__(self, token: str, chat_id: str, signal_generator, db_manager, pattern_learner):
        self.token = token
        self.chat_id = chat_id
        self.signal_generator = signal_generator
        self.db = db_manager
        self.pattern_learner = pattern_learner
        self.application = None

    async def start(self):
        """Запускает Telegram бота"""
        self.application = Application.builder().token(self.token).build()

        # Регистрируем команды
        self.application.add_handler(CommandHandler("signal", self.signal_command))
        self.application.add_handler(CommandHandler("stat", self.stat_command))
        self.application.add_handler(CommandHandler("learn", self.learn_command))

        # Запускаем бота

        await self.application.run_polling()

    async def signal_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /signal BTC"""
        if len(context.args) == 0:
            await update.message.reply_text("Usage: /signal <symbol>\nExample: /signal BTC")
            return

        symbol = context.args[0].upper()
        if not symbol.endswith('/USDT'):
            symbol = f"{symbol}/USDT"

        # Генерируем сигнал (нужно реализовать в основном цикле)
        signal_data = self._get_current_signal(symbol)

        if signal_data:
            message = self._format_signal_message(signal_data)
            await update.message.reply_text(message, parse_mode='HTML')
        else:
            await update.message.reply_text(f"No signal available for {symbol}")

    async def stat_command(self, update, context):
        try:
            arg = context.args[0] if context.args else "7d"

            days = int(arg.replace("d", ""))

            stats = self.db.get_stats(days)

            if not stats:
                await update.message.reply_text("Нет данных")
                return

            text = f"""
    📊 Статистика за {days} дней

    Сделок: {stats['total']}
    Winrate: {stats['winrate']:.2f}%
    Avg PnL: {stats['avg_pnl']:.2f}%
    Total PnL: {stats['total_pnl']:.2f}%
    """

            await update.message.reply_text(text)

        except Exception as e:
            await update.message.reply_text(f"Ошибка: {e}")

    async def learn_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Принудительное обучение"""
        await update.message.reply_text("🔄 Starting manual learning...")

        # Запускаем обучение в фоне
        asyncio.create_task(self._run_learning())

        await update.message.reply_text("✅ Learning started in background")

    def _format_signal_message(self, signal: dict) -> str:
        """Форматирует сигнал для отправки в Telegram"""
        symbol = signal['symbol'].replace('/USDT', '')
        signal_type = signal['signal_type'].upper()

        # Эмодзи для типа сигнала
        emoji = "🟢" if signal_type == "LONG" else "🔴"

        # Формируем сообщение
        message = f"""
{emoji} <b>{signal_type} {symbol}</b>
Regime: {signal['regime']} | Composite: {signal['composite']:.2f}
Bias: {signal['bias_short']} / {signal['bias_long']}

Entry: {signal['entry']:.2f}
TP: {signal['tp']:.2f} (+{signal['tp_pct']:.2f}%) [вероятность {signal['probability'] * 100:.0f}%]
SL: {signal['sl']:.2f} ({signal['sl_pct']:.2f}%)

Сигналы по TF:
"""

        # Добавляем информацию по таймфреймам
        for tf, tf_signal in signal['timeframes'].items():
            tf_emoji = "🟢" if tf_signal['signal'] == "LONG" else "🔴" if tf_signal['signal'] == "SHORT" else "🟡"
            message += f"{tf}: {tf_emoji} {tf_signal['signal']}\n"

        if signal.get('conflict'):
            message += f"\n⚠️ <b>CONFLICT DETECTED</b>: {signal['conflict_reason']}\n"

        message += f"\n📊 Статистика паттерна: +{signal['expected_move']:.1f}% за 4ч ({signal['probability'] * 100:.0f}% успеха)"

        return message

    def _get_current_signal(self, symbol):
        from pine_logic.signal_generator import SignalGenerator  # создайте этот файл

        generator = SignalGenerator(mode='SCALP')
        df = self.db.load_ohlcv(symbol, '15m')  # основной таймфрейм

        if df.empty or len(df) < 100:
            return None

        signal = generator.generate_signal(df)  # нужно реализовать

        # Добавляем предсказание из паттернов
        if hasattr(self, 'pattern_learner') and self.pattern_learner:
            prob, move = self.pattern_learner.get_prediction(
                signal['regime'], signal['composite'],
                signal['bias_short'], signal['bias_long']
            )
            signal['probability'] = prob
            signal['expected_move'] = move

        return signal

    def _get_statistics(self, days):
        return None
    def _format_stats_message(self, stats: dict, days: int) -> str:
        """Форматирует статистику для отправки"""
        message = f"""
📊 <b>Статистика за {days} день(дней)</b>

Всего сигналов: {stats['total_signals']}
Винрейт: {stats['win_rate'] * 100:.1f}% ({stats['winning']}/{stats['total_signals']})
Средний PnL: {stats['avg_pnl']:.2f}%
Общий PnL: {stats['total_pnl']:.2f}%

🏆 Лучшая сделка: +{stats['best_trade']:.2f}%
💀 Худшая сделка: {stats['worst_trade']:.2f}%
"""
        return message

    async def send_signal(self, signal_data: dict):
        """Отправляет сигнал в Telegram"""
        if not self.application:
            return

        message = self._format_signal_message(signal_data)

        try:
            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
        except Exception as e:
            print(f"Error sending Telegram message: {e}")

    async def _run_learning(self):
        """Запускает процесс обучения"""
        from ml_engine.weekly_trainer import WeeklyTrainer
        trainer = WeeklyTrainer(self.db, self.pattern_learner)
        await trainer.run_training()
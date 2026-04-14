import os
from dotenv import load_dotenv

load_dotenv()

BYBIT_API_KEY = os.getenv('BYBIT_API_KEY', '')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET', '')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Уменьшим количество символов для первого теста
SYMBOLS = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SUI/USDT:USDT', 'TAC/USDT:USDT', 'ICP/USDT:USDT', 'APT/USDT:USDT']  # Пока только 2 символа

# Таймфреймы
TIMEFRAMES = {
    '15m': 15,
    '1h': 60,
    '4h': 240
}

DB_PATH = 'database/market_data.db'
TRAIN_HOUR = 0
TRAIN_DAY = 0
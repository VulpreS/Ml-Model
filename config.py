import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False


load_dotenv()

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SUI/USDT:USDT",
    "TAC/USDT:USDT",
    "ICP/USDT:USDT",
    "APT/USDT:USDT",
]

TIMEFRAMES = {
    "15m": 15,
    "1h": 60,
    "4h": 240,
}

DB_PATH = "database/market_data.db"
ML_MODEL_PATH = "ml_model.pkl"
TRAIN_HOUR = 0
TRAIN_DAY = 0

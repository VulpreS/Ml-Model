# reset_db.py
import os
if os.path.exists('database/market_data.db'):
    os.remove('database/market_data.db')
    print("Database deleted")
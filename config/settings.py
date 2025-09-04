import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Alpaca API Configuration
    APCA_API_KEY_ID = os.getenv('APCA_API_KEY_ID')
    APCA_API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
    APCA_API_BASE_URL = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
    
    # ... rest of your settings

settings = Settings()  # ← This line is crucial

from datetime import datetime, timedelta
import os 

# Time in seconds after which files will be automatically deleted
FILE_DELETION_TIME = 3600  # 1 hour
TIME_WINDOW = timedelta(minutes=1)

# Global variable to store the current model
current_model = "auto"

# Rate Limiting
MAX_REQUESTS = 5
TIME_WINDOW = timedelta(minutes=1)

middle_value = 128
low_value_threshold = 0.03
high_value_threshold = 0.97

from dotenv import load_dotenv
load_dotenv()
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

DEVELOPER_PASSWORD = "HADIE"



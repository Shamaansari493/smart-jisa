import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


APP_NAME = "smart_jisa_app"
DEFAULT_USER_ID = "default_user"
DEFAULT_SESSION_ID = "default_session"

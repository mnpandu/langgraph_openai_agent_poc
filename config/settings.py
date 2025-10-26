import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("❌ Missing OPENAI_API_KEY in .env file")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

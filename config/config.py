import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    API_KEY = os.environ.get("API_KEY")
    DEBUG = os.environ.get("DEBUG", "False") == "True"
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    MODEL_DOWNLOAD_ID = os.environ.get("MODEL_DOWNLOAD_ID")
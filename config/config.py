import os

class Config:
    API_KEY = os.environ.get("API_KEY")
    DEBUG = os.environ.get("DEBUG", "False") == "True"
    MODEL_DOWNLOAD_ID = os.environ.get("MODEL_DOWNLOAD_ID")
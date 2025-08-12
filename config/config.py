import os

class Config:
    API_KEY = os.environ.get("API_KEY")
    DEBUG = os.environ.get("DEBUG", "False") == "True"
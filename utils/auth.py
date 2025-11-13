from config.config import Config


def is_authenticated(api_key: str) -> bool:
    return isinstance(api_key, str) and api_key == Config.API_KEY
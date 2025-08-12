from email import header
import re
from datetime import datetime

def is_iso8601(s):
    """
    Check if the string is a valid ISO 8601 timestamp.
    """
    
    iso8601_regex = re.compile(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$"
    )
    
    if not isinstance(s, str):
        return False
    if not iso8601_regex.match(s):
        return False
    
    try:
        datetime.fromisoformat(s.replace("Z", "+00:00"))
        return True
    except Exception:
        return False
    
def validate_payload(header, payload):
    """
    Validate JSON payload from /predict
    """
    
    required_keys = {"X-Api-Key", "Date"}
    if not isinstance(header, dict) or not required_keys.issubset(header):
        return False, "Missing required fields"

    if not isinstance(header["X-Api-Key"], str):
        return False, "API key must be a string"

    if not is_iso8601(header["Date"]):
        return False, "Date must be a valid ISO 8601 string"
    
    required_video_keys = {"encoding", "format", "content", "fps"}
    if not isinstance(payload, dict) or not all(key in payload for key in required_video_keys):
        return False, "video_input is missing required fields"
    
    if payload["encoding"] != "base64":
        return False, "video encoding must be 'base64'"

    if not isinstance(payload["format"], str) or payload["format"] != "mp4":
        return False, "video format must be 'mp4'"

    if not isinstance(payload["content"], str):
        return False, "video content must be base64 string"
    
    if not isinstance(payload["fps"], (int, float)) or payload["fps"] <= 0:
        return False, "video fps must be a positive number"

    return True, None
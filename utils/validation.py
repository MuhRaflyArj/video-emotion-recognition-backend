import re
from datetime import datetime


ISO_8601_REGEX = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$"
)


def is_iso8601(value):
    if not isinstance(value, str) or not ISO_8601_REGEX.match(value):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def validate_headers(header):
    required_header_keys = {"X-Api-Key", "Date"}
    if not isinstance(header, dict) or not required_header_keys.issubset(header):
        return False, "Missing required headers"

    if not isinstance(header["X-Api-Key"], str) or not header["X-Api-Key"]:
        return False, "Header X-Api-Key must be a non-empty string"

    if not is_iso8601(header["Date"]):
        return False, "Header Date must be a valid ISO 8601 timestamp"

    return True, None


def validate_predict_payload(header, payload):
    is_valid, error = validate_headers(header)
    if not is_valid:
        return False, error

    required_payload_keys = {"url", "format", "fps"}
    if not isinstance(payload, dict) or not required_payload_keys.issubset(payload):
        return False, "Payload must include 'url', 'format', and 'fps' fields"

    if not isinstance(payload["url"], str) or not payload["url"].startswith("https://storage.googleapis.com/"):
        return False, "Video URL must be a Google Cloud Storage public URL"

    if payload["format"].lower() != "mp4":
        return False, "Video format must be 'mp4'"

    if not isinstance(payload["fps"], (int, float)) or payload["fps"] <= 0:
        return False, "Video fps must be a positive number"

    return True, None


def validate_thumbnail_payload(header, payload):
    is_valid, error = validate_headers(header)
    if not is_valid:
        return False, error

    required_payload_keys = {"url", "format", "fps", "user_id", "journal_id"}
    if not isinstance(payload, dict) or not required_payload_keys.issubset(payload):
        return False, "Payload must include 'url', 'format', 'fps', 'user_id', and 'journal_id' fields"

    if not isinstance(payload["url"], str) or not payload["url"].startswith("https://storage.googleapis.com/"):
        return False, "Video URL must be a Google Cloud Storage public URL"

    if payload["format"].lower() != "mp4":
        return False, "Video format must be 'mp4'"

    if not isinstance(payload["fps"], (int, float)) or payload["fps"] <= 0:
        return False, "Video fps must be a positive number"

    for field in ("user_id", "journal_id"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            return False, f"{field} must be a non-empty string"

    return True, None
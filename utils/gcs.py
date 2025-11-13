from urllib.parse import unquote, urlparse
import io

from google.api_core import exceptions as gcloud_exceptions
from google.cloud import storage


def parse_gcs_url(url: str):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "storage.googleapis.com":
        raise ValueError("video url must point to storage.googleapis.com")
    path = parsed.path.lstrip("/")
    if not path or "/" not in path:
        raise ValueError("invalid Google Cloud Storage URL format")
    bucket_name, blob_name = path.split("/", 1)
    return bucket_name, unquote(blob_name)


def download_gcs_bytes(bucket_name, blob_name, timeout=None):
    try:
        client = storage.Client.create_anonymous_client()
        blob = client.bucket(bucket_name).blob(blob_name)
        return blob.download_as_bytes(timeout=timeout), None
    except gcloud_exceptions.NotFound:
        return None, "video object not found in Google Cloud Storage"
    except gcloud_exceptions.GoogleAPIError as exc:
        return None, f"Google Cloud Storage error: {exc}"
    except Exception as exc:
        return None, f"Unexpected error: {exc}"


def upload_gcs_bytes(bucket_name, blob_name, data, timeout=None, content_type="video/mp4"):
    try:
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_name)
        blob.cache_control = "public, max-age=3600"
        blob.upload_from_file(io.BytesIO(data), timeout=timeout, content_type=content_type, rewind=True)
        blob.make_public()
        return None
    except gcloud_exceptions.GoogleAPIError as exc:
        return f"Google Cloud Storage upload error: {exc}"
    except Exception as exc:
        return f"Unexpected upload error: {exc}"


def fetch_video_bytes(url, timeout=(5, 30)):
    bucket_name, blob_name = parse_gcs_url(url)
    return download_gcs_bytes(bucket_name, blob_name, timeout=max(timeout))
import uuid

def generate_filename(suffix: str = ".mp4") -> str:
    return f"{uuid.uuid4().hex}{suffix}"
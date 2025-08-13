import os
import gdown
from flask import current_app

def download_model(model_dir="models", model_filename="model.pth"):
    """
    Download the model from Google Drive if not already present
    """
    
    try:
        model_path = os.path.join(model_dir, model_filename)
        download_id = current_app.config.get("MODEL_DOWNLOAD_ID")

        if not download_id:
            return None, "MODEL ID is invalid"

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if not os.path.exists(model_path):
            url = f"https://drive.google.com/uc?id={download_id}"
            gdown.download(url, model_path, quiet=False)

        return model_path, None
    
    except Exception as e:
        return None, str(e)
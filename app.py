from flask import Flask, request, jsonify
import time

from config.config import Config
from utils.validation import validate_payload
from utils.video import decode_video, extract_facemesh
from models.downloader import download_model
from models.predictor import predict_emotion


app = Flask(__name__)
app.config.from_object(Config)

def authenticated(api_key):
    return api_key == app.config['API_KEY']

@app.route('/predict', methods=['POST'])
def predict():
    start = time.perf_counter()
    try: 
        header = dict(request.headers)
        payload = request.get_json()
        
        print(header)
        
        print(payload)
        
        valid_payload, error = validate_payload(header, payload)
        if not valid_payload:
            return jsonify({"error": error}), 400
        
        api_key = header.get('X-Api-Key')
        if not authenticated(api_key):
            return jsonify({"error": "Unauthorized"}), 401

        payload = request.get_json()
        if not payload:
            return jsonify({"error": "Missing or invalid JSON payload"}), 400

        video_bytes, err = decode_video(payload["content"])
        if err:
            return jsonify({"error": err}), 400

        arr, err = extract_facemesh(video_bytes, container_format=payload["format"])
        if err:
            return jsonify({"error": err}), 400
        
        model_path, err = download_model(model_filename="EmotiMesh_Net.pth")
        if err:
            return jsonify({"error": err}), 500
        
        pred_class = predict_emotion(arr, model_path=model_path)

        class_labels = ["Anger", "Happy", "Shock", "Neutral", "Sad"]
        predicted_label = class_labels[pred_class] if 0 <= pred_class < len(class_labels) else "Unknown"

        return jsonify({
            "prediction": predicted_label,
            "latency_ms": int((time.perf_counter() - start) * 1000)
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
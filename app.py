import base64
from flask import Flask, request, jsonify
import time

from config.config import Config
from logutils.logger import get_logs, log_request
from utils.validation import validate_payload
from utils.video import decode_video, extract_facemesh, generate_video
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
        
        # Check for payload
        if not payload:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(400, latency_ms, False, error_message="Missing or invalid JSON payload")
            return jsonify({"error": "Missing or invalid JSON payload"}), 400
        
        # Validate API key
        api_key = header.get('X-Api-Key')
        if not authenticated(api_key):
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(401, latency_ms, False, error_message="Unauthorized")
            return jsonify({"error": "Unauthorized"}), 401

        # Decode video to mp4
        video_bytes, err = decode_video(payload["content"])
        if err:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(400, latency_ms, False, error_message="Missing or invalid video content")
            return jsonify({"error": "Missing or invalid video content"}), 400

        # Extract face vectors from video
        arr, err = extract_facemesh(video_bytes, container_format=payload["format"])
        if err:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(400, latency_ms, False, error_message=err)
            return jsonify({"error": err}), 400
        
        # Download model if the model not exist
        model_path, err = download_model(model_filename="EmotiMesh_Net.pth")
        if err:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(500, latency_ms, False, error_message=err)
            return jsonify({"error": err}), 500
        
        # Predict the emotion based on facemesh vectors
        pred_class, confidence = predict_emotion(arr, model_path=model_path)

        class_labels = ["Anger", "Happy", "Shock", "Neutral", "Sad"]
        predicted_label = class_labels[pred_class] if 0 <= pred_class < len(class_labels) else "Unknown"

        latency_ms = int((time.perf_counter() - start) * 1000)
        log_request(200, latency_ms, True, prediction=predicted_label, confidence=round(confidence, 4))
        return jsonify({
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "latency_ms": latency_ms
        }), 200
        
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        log_request(500, latency_ms, False, error_message=f"Internal server error: {str(e)}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500
    
    
@app.route('/logs', methods=['GET'])
def logs():
    start = time.perf_counter()
    
    try:
        header = dict(request.headers)
        
        # Validate API key
        api_key = header.get('X-Api-Key')
        if not authenticated(api_key):
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(401, latency_ms, False, error_message="Unauthorized")
            return jsonify({
                "error": "Unauthorized"
            }), 401

        # Check for filters and get logs based on filters
        filters = request.get_json()
        logs_data = get_logs(filters)
        
        latency_ms = int((time.perf_counter() - start) * 1000)
        log_request(200, latency_ms, True)
        return jsonify({
            "logs": logs_data,
            "count": len(logs_data),
            "latency_ms": latency_ms
        }), 200
        
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        log_request(500, latency_ms, False, error_message=f"Internal server error: {str(e)}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/thumbnail', methods=['POST'])
def thumbnail():
    start = time.perf_counter()
    try:
        header = dict(request.headers)
        payload = request.get_json()
        
        # Validate payload structure
        if not payload:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(400, latency_ms, False, error_message="Missing or invalid JSON payload")
            return jsonify({"error": "Missing or invalid JSON payload"}), 400
        
        # Validate API key
        api_key = header.get('X-Api-Key')
        if not authenticated(api_key):
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(401, latency_ms, False, error_message="Unauthorized")
            return jsonify({"error": "Unauthorized"}), 401
        
        # Check required fields
        if 'content' not in payload:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(400, latency_ms, False, error_message="Missing 'content' field")
            return jsonify({"error": "Missing 'content' field"}), 400
        
        # Get optional parameters
        container_format = payload.get('format', 'mp4')
        target_duration = payload.get('duration', 3.0)
        
        # Validate duration
        if not isinstance(target_duration, (int, float)) or target_duration <= 0 or target_duration > 10:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(400, latency_ms, False, error_message="Duration must be between 0.1 and 10 seconds")
            return jsonify({"error": "Duration must be between 0.1 and 10 seconds"}), 400
        
        # Decode video
        video_bytes, err = decode_video(payload["content"])
        if err:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(400, latency_ms, False, error_message="Missing or invalid video content")
            return jsonify({"error": str(err)}), 400

        # Generate video
        generated_video, err = generate_video(video_bytes, container_format=container_format, target_duration=target_duration)
        if err:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(400, latency_ms, False, error_message=err)
            return jsonify({"error": err}), 400
        
        # Convert video to base64
        video_b64 = base64.b64encode(generated_video).decode('utf-8')
        
        latency_ms = int((time.perf_counter() - start) * 1000)
        log_request(200, latency_ms, True)
        
        return jsonify({
            "video": video_b64,
            "latency_ms": latency_ms,
            "duration": target_duration
        }), 200
        
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        log_request(500, latency_ms, False, error_message=f"Internal server error: {str(e)}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
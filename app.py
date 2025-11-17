import time

from dotenv import load_dotenv
from flask import Flask, request, jsonify

from config.config import Config
from logutils.logger import get_logs, log_request
from utils.validation import (
    validate_predict_payload,
    validate_thumbnail_payload,
)
from utils.video import extract_face_images, generate_video
from models.downloader import download_model
from models.predictor import predict_emotion
from utils.auth import is_authenticated
from utils.files import generate_filename
from utils.gcs import fetch_video_bytes, upload_gcs_bytes

load_dotenv()

app = Flask(__name__)
app.config.from_object(Config)


@app.route('/predict', methods=['POST'])
def predict():
    start = time.perf_counter()
    try:
        header = dict(request.headers)
        payload = request.get_json()

        if not payload:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(400, latency_ms, False, error_message="Empty payload")
            return jsonify({"error": "Empty payload"}), 400

        is_valid, validation_error = validate_predict_payload(header, payload)
        if not is_valid:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(400, latency_ms, False, error_message=validation_error)
            return jsonify({"error": validation_error}), 400

        api_key = header.get("X-Api-Key")
        if not is_authenticated(api_key):
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(401, latency_ms, False, error_message="Invalid API key")
            return jsonify({"error": "Unauthorized"}), 401

        video_bytes, err = fetch_video_bytes(payload["url"])
        if err:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(502, latency_ms, False, error_message=f"Video download failed: {err}")
            return jsonify({"error": "Failed to download video content"}), 502

        image_sequence, err = extract_face_images(video_bytes, container_format=payload["format"])
        if err:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(422, latency_ms, False, error_message=f"Face extraction failed: {err}")
            return jsonify({"error": "Could not extract faces from video"}), 422

        model_path, err = download_model(model_filename="EfficientNetV2S.pth")
        if err:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(500, latency_ms, False, error_message=f"Model download failed: {err}")
            return jsonify({"error": f"Failed to load model {err}"}), 500

        pred_class, confidence = predict_emotion(image_sequence, model_path=model_path)
        class_labels = ["Anger", "Happy", "Neutral", "Sad", "Shock"]
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
        log_request(500, latency_ms, False, error_message=f"Internal server error: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/thumbnail', methods=['POST'])
def thumbnail():
    start = time.perf_counter()
    try:
        header = dict(request.headers)
        payload = request.get_json()

        if not payload:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(400, latency_ms, False, error_message="Empty payload")
            return jsonify({"error": "Empty payload"}), 400

        is_valid, validation_error = validate_thumbnail_payload(header, payload)
        if not is_valid:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(400, latency_ms, False, error_message=validation_error)
            return jsonify({"error": validation_error}), 400

        api_key = header.get("X-Api-Key")
        if not is_authenticated(api_key):
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(401, latency_ms, False, error_message="Invalid API key")
            return jsonify({"error": "Unauthorized"}), 401

        video_bytes, err = fetch_video_bytes(payload["url"])
        if err:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(502, latency_ms, False, error_message=f"Video download failed: {err}")
            return jsonify({"error": "Failed to download video content"}), 502

        generated_video, err = generate_video(
            video_bytes,
            container_format=payload["format"],
            target_fps=payload["fps"]
        )
        if err:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(422, latency_ms, False, error_message=f"Thumbnail generation failed: {err}")
            return jsonify({"error": "Failed to generate thumbnail video"}), 422

        bucket_name = app.config.get("BUCKET_NAME")
        if not bucket_name:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(500, latency_ms, False, error_message="Missing BUCKET_NAME configuration")
            return jsonify({"error": "Server misconfiguration"}), 500

        filename = generate_filename()
        gcs_object_path = (
            f"uploads/videos/{payload['user_id']}/"
            f"{payload['journal_id']}/recordings/{filename}"
        )

        upload_err = upload_gcs_bytes(bucket_name, gcs_object_path, generated_video)
        if upload_err:
            latency_ms = int((time.perf_counter() - start) * 1000)
            log_request(502, latency_ms, False, error_message=upload_err)
            return jsonify({"error": f"Failed to store thumbnail, {upload_err}"}), 502

        public_uri = f"https://storage.googleapis.com/{bucket_name}/{gcs_object_path}"
        latency_ms = int((time.perf_counter() - start) * 1000)
        log_request(200, latency_ms, True)

        return jsonify({
            "thumbnail_uri": public_uri,
            "latency_ms": latency_ms
        }), 200

    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        log_request(500, latency_ms, False, error_message=f"Internal server error: {e}")
        return jsonify({"error": "Internal server error"}), 500
        
@app.route('/logs', methods=['GET'])
def logs():
    start = time.perf_counter()
    
    try:
        header = dict(request.headers)
        
        # Validate API key
        api_key = header.get('X-Api-Key')
        if not is_authenticated(api_key):
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
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)

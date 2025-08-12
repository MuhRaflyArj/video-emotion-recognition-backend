from flask import Flask, request, jsonify
from config.config import Config
from utils.validation import validate_payload
from utils.video import decode_video, extract_facemesh
import time

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

        # For demonstration, return the shape of the numpy array (not the array itself)
        return jsonify({
            "facemesh_shape": list(arr.shape),
            "facemesh_vectors": arr.tolist(),
            "latency_ms": int((time.perf_counter() - start) * 1000)
        }), 200
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
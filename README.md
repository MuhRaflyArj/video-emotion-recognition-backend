

<h1 align="center">🖼️ Video Emotion Recognition API for Journaling Thumbnails 🤖</h1>

<h1 align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python" alt="Python 3.11">
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/MediaPipe-007BFF?style=for-the-badge&logo=mediapipe&logoColor=white" alt="MediaPipe">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
</h1>

## 🎯 Goal

This backend service provides an API for generating expressive video thumbnails for a journaling application. Users upload a short video, and the API predicts the emotion shown in the video, returning both the predicted emotion and a "boomerang" thumbnail video. This enables journaling apps to visually summarize daily moods with an emotion-aware video cover.

<p align="center">
  <img src="./assets/thumbnail.gif" width="500" alt="Generated Thumbnail Demo">
  <br>
  <em>*Example of a generated thumbnail video*</em>
</p>

---

## 🚀 Backend Features

- **🎭 Predict 5 Emotions:**  
  Classifies uploaded video clips into one of five emotions: `Anger`, `Happy`, `Shock`, `Neutral`, or `Sad` using a deep learning model based on MediaPipe Face Mesh and a CNN-LSTM architecture.

- **🎬 Generate Thumbnail Video:**  
  Creates a short, looping "boomerang" video from the uploaded clip, suitable for use as a dynamic thumbnail in journaling apps.

- **🔐 API Key Authentication:**  
  All endpoints require a valid API key for access, ensuring secure usage.

- **📝 Logging with Filtering:**  
  Every request is logged with details such as timestamp, client ID, endpoint, prediction, confidence, and latency for monitoring and analytics.  
  Logs can be filtered by request parameters.

---

## 🛠️ Tech Stack

- **Python 3.11**
- **Flask** (REST API)
- **MediaPipe** (Face Mesh extraction)
- **PyTorch** (Emotion classification model)
- **OpenCV & PyAV** (Video processing)
- **Docker** (Containerization)

---
## 🐳 Deployment

The following steps have been tested on **Debian 12 (Bookworm)** running on Google Cloud Platform (GCP) with an **e2-standard-2** VM and **25GB disk**.

**The API will be available on port `5000`.**

### 1. Install Git and Docker

```sh
sudo apt-get update
sudo apt-get install -y git docker.io
```

### 2. Clone the Repository

```sh
git clone https://github.com/MuhRaflyArj/video-emotion-recognition-backend.git
cd video-emotion-recognition-backend
```

### 3. Prepare Environment Variables

Copy the example environment file and edit it:

```sh
cp .env.example .env
```

- **Set your API key**.
- **Set the Google Drive model ID**:
  - Upload your trained model (e.g., `EmotiMesh_Net.pth`) to Google Drive.
  - Make sure the file is shared publicly (anyone with the link can view/download).
  - Copy the file ID from the shareable link (the part after `id=` in the URL).
  - Paste this ID as the value for `MODEL_DOWNLOAD_ID` in your `.env` file.

Example `.env`:
```
API_KEY=your_api_key_here
DEBUG=False
MODEL_DOWNLOAD_ID=your_gdrive_file_id_here
```

### 4. Run the Initialization Script

This will build the Docker image and start the API container on port 5000.

```sh
chmod +x init.sh
./init.sh
```

The API will be accessible at:  
`http://<external-server-ip>:5000`

---

## 📦 Endpoints

### `POST /predict`

**Description:**  
Upload a base64-encoded video and receive the predicted emotion and confidence.

**Headers:**
- `x-api-key`: Your API key
- `date`: Current date and time in ISO 8601 format (e.g., `2025-08-16T12:34:56Z`)

**Payload Example:**
```json
{
  "encoding": "base64",
  "format": "mp4",
  "fps": 30,
  "content": "<base64 string>"
}
```

**Response Example:**
```json
{
  "emotion": "Happy",
  "confidence": 0.92
}
```

---

### `POST /thumbnail`

**Description:**  
Upload a base64-encoded video and receive a generated "boomerang" thumbnail video (base64-encoded).

**Headers:**
- `x-api-key`: Your API key
- `date`: Current date and time in ISO 8601 format (e.g., `2025-08-16T12:34:56Z`)

**Payload Example:**
```json
{
  "encoding": "base64",
  "format": "mp4",
  "fps": 30,
  "content": "<base64 string>"
}
```

**Response Example:**
```json
{
  "thumbnail": "<base64 string>"
}
```

---

### `GET /logs`

**Description:**  
Retrieve logs of previous API requests. Requires authentication.  
Supports filtering by various parameters via JSON body in the request.

**Headers:**
- `x-api-key`: Your API key
- `date`: Current date and time in ISO 8601 format (e.g., `2025-08-16T12:34:56Z`)

**Filtering Options (send as JSON body):**
- `start_date`: (string, ISO format) Filter logs from this date/time.
- `end_date`: (string, ISO format) Filter logs up to this date/time.
- `status_code`: (string or int) Filter by HTTP status code (e.g., `"200"`).
- `success`: (string `"True"` or `"False"`) Filter by request success.
- `client_id`: (string) Filter by client ID.

**Examples:**

- **All logs:**
  ```json
  {}
  ```

- **By date range:**
  ```json
  {
    "start_date": "2025-08-13T00:00:00",
    "end_date": "2025-08-13T23:59:59"
  }
  ```

- **By status code:**
  ```json
  {
    "status_code": "200"
  }
  ```

- **By success:**
  ```json
  {
    "success": "True"
  }
  ```

- **By client ID:**
  ```json
  {
    "client_id": "your_client_id_here"
  }
  ```

- **Combined filters:**
  ```json
  {
    "start_date": "2025-08-13T00:00:00",
    "end_date": "2025-08-13T23:59:59",
    "success": "False",
    "client_id": "your_client_id_here"
  }
  ```

---

## 📚 Model

The backend uses the [`EmotiMesh_Net`](models/predictor.py) model, trained on AFEW-VA dataset to recognize emotions from 3D facial landmarks extracted by MediaPipe.

> **Model Training:**  
> You can train or fine-tune the model yourself using the [video-based-emotion-recognition](https://github.com/MuhRaflyArj/video-based-emotion-recognition) repository (also maintained by the author of this backend).  
> That repository contains the full training pipeline, dataset processing, and model development details.

---
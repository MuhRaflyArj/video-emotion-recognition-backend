import io
import math
import os
import tempfile

import av
import cv2
import mediapipe as mp
import numpy as np

def extract_frames(video_bytes, container_format="mp4"):
    """
    Extract all frames from a byte video stream.
    """
    try:
        container = av.open(io.BytesIO(video_bytes), format=container_format)
    except av.AVError as exc:
        return None, f"Unable to open video container: {exc}"
    except Exception as exc:
        return None, f"Unexpected error opening video: {exc}"

    frames = []
    try:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="bgr24"))
    except av.AVError as exc:
        container.close()
        return None, f"Failed to decode frames: {exc}"
    except Exception as exc:
        container.close()
        return None, f"Unexpected error decoding frames: {exc}"

    container.close()

    if not frames:
        return None, "No video frames found"

    return frames, None


def extract_facemesh(video_bytes, container_format="mp4"):
    """
    Extract MediaPipe facemesh vectors from the video.
    Returns (rows, error_message)
    """
    frames, err = extract_frames(video_bytes, container_format=container_format)
    if err:
        return None, err

    total_frames = len(frames)
    if total_frames < 15:
        return None, "Video too short for facemesh extraction (needs >= 15 frames)"

    indices = [int(round(i * (total_frames - 1) / 14)) for i in range(15)]
    rows = []
    len_vector = 1404  # 468 points * 3 (x,y,z)

    mp_face_mesh = mp.solutions.face_mesh
    try:
        with mp_face_mesh.FaceMesh(static_image_mode=False,
                                   refine_landmarks=True,
                                   max_num_faces=1,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5) as face_mesh:

            for idx in indices:
                frame = frames[idx]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb_frame)

                if not result.multi_face_landmarks:
                    rows.append(np.zeros(len_vector, dtype=np.float32))
                    continue

                landmarks = result.multi_face_landmarks[0].landmark
                vector = []
                for lm in landmarks:
                    vector.extend([lm.x, lm.y, lm.z])
                rows.append(np.array(vector, dtype=np.float32))

    except Exception as exc:
        return None, f"Facemesh extraction failed: {exc}"

    if not rows:
        return None, "Failed to extract facemesh data"

    return np.stack(rows, axis=0), None


def get_face_bbox(frame_shape, relative_bbox, margin=0.1):
    """Calculate bounding box from relative coordinates with margin."""
    height, width, _ = frame_shape

    if hasattr(relative_bbox, "xmin"):
        xmin = relative_bbox.xmin
        ymin = relative_bbox.ymin
        box_width = relative_bbox.width
        box_height = relative_bbox.height
    else:
        xmin, ymin, box_width, box_height = relative_bbox

    xmin = max(xmin, 0.0)
    ymin = max(ymin, 0.0)
    xmax = min(xmin + box_width, 1.0)
    ymax = min(ymin + box_height, 1.0)

    x1 = int((xmin - margin * box_width) * width)
    y1 = int((ymin - margin * box_height) * height)
    x2 = int((xmax + margin * box_width) * width)
    y2 = int((ymax + margin * box_height) * height)

    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, width - 1)
    y2 = min(y2, height - 1)

    if x2 <= x1 or y2 <= y1:
        return 0, 0, width, height

    return x1, y1, x2, y2


def extract_face_images(video_bytes,
                        container_format="mp4",
                        num_images=18,
                        face_img_size=(224, 224)):
    """
    Extract evenly spaced face crops from the video and return as a tensor-like numpy array.
    Output shape: (1, num_images, H, W)
    """
    frames, err = extract_frames(video_bytes, container_format=container_format)
    if err:
        return None, err

    total_frames = len(frames)
    if total_frames < num_images:
        return None, f"Video must contain at least {num_images} frames"

    indices = np.linspace(0, total_frames - 1, num_images, dtype=int)

    mp_face_detection = mp.solutions.face_detection
    crops = []

    try:
        with mp_face_detection.FaceDetection(model_selection=1,
                                             min_detection_confidence=0.5) as detector:
            for idx in indices:
                frame = frames[idx]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.process(rgb_frame)

                if result.detections:
                    detection = result.detections[0]
                    bbox = detection.location_data.relative_bounding_box
                    x1, y1, x2, y2 = get_face_bbox(frame.shape, bbox)
                else:
                    # fallback to full frame if no face detected
                    x1, y1, x2, y2 = 0, 0, frame.shape[1] - 1, frame.shape[0] - 1

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    face = frame

                face = cv2.resize(face, face_img_size, interpolation=cv2.INTER_AREA)
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                crops.append(face)

    except Exception as exc:
        return None, f"Face extraction failed: {exc}"

    if len(crops) < num_images:
        return None, "Could not extract enough face crops"

    stack = np.stack(crops, axis=0).astype(np.float32) / 255.0
    stack = np.expand_dims(stack, axis=0)  # shape (1, num_images, H, W)

    return stack, None

def generate_video(video_bytes,
                   container_format="mp4",
                   target_fps=10,
                   target_duration=3.0):
    frames, err = extract_frames(video_bytes, container_format=container_format)
    if err:
        return None, err

    if target_fps <= 0 or target_duration <= 0:
        return None, "target_fps and target_duration must be positive"

    max_frames = max(1, int(math.ceil(target_fps * target_duration)))
    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        selected = [frames[i] for i in indices]
    else:
        selected = frames

    height, width, _ = selected[0].shape
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_path = temp_file.name
    temp_file.close()

    try:
        output = av.open(temp_path, mode="w", format="mp4", options={"movflags": "+faststart"})
        stream = output.add_stream("h264", rate=target_fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        for frame in selected:
            av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            packet = stream.encode(av_frame)
            if packet:
                output.mux(packet)

        packet = stream.encode(None)
        if packet:
            output.mux(packet)

        output.close()

        with open(temp_path, "rb") as fh:
            data = fh.read()
        return data, None
    except Exception as exc:
        return None, f"Failed to generate video: {exc}"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
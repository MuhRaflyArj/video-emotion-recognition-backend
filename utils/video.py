import base64
import io
import numpy as np
import mediapipe as mp
import av

def decode_video(b64_str):
    """
    Decode base64 video to bytes
    """
    
    try:
        s = b64_str.strip()
        if s.startswith("data:"):
            s = s.split(",", 1)[1]
        return base64.b64decode(s, validate=True), None
    except Exception:
        return None, "Invalid base64 video content"

def extract_facemesh(video_bytes, container_format="mp4"):
    """
    Extract mediapipe face vectors from video
    """
    
    try:
        bio = io.BytesIO(video_bytes)
        try:
            container = av.open(bio, format=container_format)
        except av.AVError:
            bio.seek(0)
            container = av.open(bio)
    except Exception as e:
        return None, f"Failed to open video, {e}"
    
    video_stream = next((s for s in container.streams if s.type == 'video'), None)
    if video_stream is None:
        container.close()
        return None, "No video found"
    
    all_frames = []
    try:
        for packet in container.demux(video_stream):
            for f in packet.decode():
                if f is not None:
                    all_frames.append(f)
                    
    except Exception as e:
        container.close()
        return None, f"Failed to decode video frames, {e}"
    
    total_frames = len(all_frames)    
    print(total_frames) # <-- Debug
    
    if total_frames < 15:
        container.close()
        return None, "Video must have at least 15 frames"
    
    indices = [int(round(i * (total_frames - 1) / 14)) for i in range(15)]
    
    rows = []
    last_row = None
    len_vector = 1404
    
    # Load facemesh
    mp_face_mesh = mp.solutions.face_mesh
    
    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35
        ) as face_mesh:
            
            for idx in indices:
                frame = all_frames[idx]
                rgb = frame.to_ndarray(format='rgb24')
                result = face_mesh.process(rgb)
                
                if result.multi_face_landmarks:
                    face_landmark = result.multi_face_landmarks[0]
                    row = []
                    
                    for point in face_landmark.landmark:
                        row.extend([point.x, point.y, point.z])
                    last_row = row
                    
                else:
                    row = last_row[:] if last_row is not None else [0.0] * len_vector

                rows.append(row)

    except Exception as e:
        container.close()
        return None, f"Failed to extract face mesh: {e}"
    
    finally:
        container.close()

    return np.asarray(rows, dtype=np.float32), None
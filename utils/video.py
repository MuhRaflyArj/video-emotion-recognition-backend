import base64
import io
import numpy as np
import mediapipe as mp
import av
import cv2

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
    
def extract_frames(video_bytes, container_format='mp4'):
    """
    Extract all frames from a byte video stream
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
            for frame in packet.decode():
                if frame is not None:
                    all_frames.append(frame)
    except Exception as e:
        container.close()
        return None, f"Failed to decode video frames, {e}"

    container.close()
    return all_frames, None

def extract_facemesh(video_bytes, container_format="mp4"):
    """
    Extract mediapipe face vectors from video
    """
    
    
    all_frames, err = extract_frames(video_bytes, container_format=container_format)
    if err:
        return None, err

    total_frames = len(all_frames)    
    if total_frames < 15:
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
        return None, f"Failed to extract face mesh: {e}"

    return np.asarray(rows, dtype=np.float32), None

def generate_video(video_bytes, container_format='mp4', target_fps=10, target_duration=3.0):
    """
    Create a boomerang effect from video
    """
    
    all_frames, err = extract_frames(video_bytes, container_format)
    if err:
        return None, err

    if len(all_frames) < 15:
        return None, "Video must have at least 15 frames"

    # Conver video to target_fps
    half_frames = int((target_duration / 2) * target_fps)
    if half_frames < int(target_duration / 2 * target_fps):
        return None, "Video is too short"
    
    indices = np.linspace(0, len(all_frames) - 1, num=half_frames, dtype=int)
    sampled_frames = [all_frames[i] for i in indices]
    
    # Generate video
    video_frames = sampled_frames + sampled_frames[-2::-1]
    
    first_frame_np = video_frames[0].to_ndarray(format='rgb24')
    height, width = first_frame_np.shape[:2]
    
    try:
        # Video container
        output_bio = io.BytesIO()
        output_container = av.open(output_bio, mode='w', format='mp4')
        
        # Add video stream
        video_stream = output_container.add_stream('h264', rate=target_fps)
        video_stream.width = width
        video_stream.height = height
        video_stream.pix_fmt = 'yuv420p'
        
        # Write frames
        for frame in video_frames:
            rgb_array = frame.to_ndarray(format='rgb24')
            new_frame = av.VideoFrame.from_ndarray(rgb_array, format='rgb24')
            
            for packet in video_stream.encode(new_frame):
                output_container.mux(packet)
                
        for packet in video_stream.encode():
            output_container.mux(packet)
            
        output_container.close()
        
        output_bio.seek(0)
        video_data = output_bio.read()
        
        return video_data, None
    
    except Exception as e:
        return None, f"Failed to generate video: {e}"
                

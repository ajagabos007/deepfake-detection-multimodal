
# --- src/preprocess/video.py ---
"""
Video preprocessing utilities:
- extract_frames: extract frames at a given rate
- detect_and_crop_faces: detect faces using OpenCV Haar cascade and save crops
"""

from pathlib import Path
import cv2
import os

HAAR_MODEL_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'


def extract_frames(video_path: str, out_dir: str, fps: int = 1):
    """Extract frames from video at `fps` frames per second.
    Saves frames as JPEG in out_dir.
    Returns list of saved frame paths.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = vidcap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = int(round(video_fps / fps))

    saved = []
    count = 0
    saved_idx = 0
    while True:
        ret, frame = vidcap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            out_path = os.path.join(out_dir, f"frame_{saved_idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved.append(out_path)
            saved_idx += 1
        count += 1

    vidcap.release()
    return saved


def detect_and_crop_faces(frame_path: str, out_dir: str, scale_factor=1.1, min_neighbors=5):
    """Detect faces in a frame and save cropped faces to out_dir. Returns list of crops."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img = cv2.imread(frame_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(HAAR_MODEL_PATH)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    crops = []
    for i, (x, y, w, h) in enumerate(faces):
        pad = int(0.2 * max(w, h))
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, img.shape[1])
        y2 = min(y + h + pad, img.shape[0])
        crop = img[y1:y2, x1:x2]
        out_path = os.path.join(out_dir, f"{Path(frame_path).stem}_face_{i}.jpg")
        cv2.imwrite(out_path, crop)
        crops.append(out_path)
    return crops


# --- README.md (brief) ---
# Deepfake Detection - Multimodal

This repo contains starting code for preprocessing video, audio, and context (transcripts).

Start by installing requirements in a virtualenv and ensure `ffmpeg` is installed system-wide.

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/predict.py path/to/video.mp4
```
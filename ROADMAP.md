# Deepfake Detection (Multi-Modal)

## Overview
Enhancing deepfake detection by integrating **visual**, **audio**, and **contextual** features.

## Current Status
- ✅ Environment setup (requirements.txt fixed for Python 3.13).
- ✅ Preprocessing implemented:
  - Video frame extraction + face detection (`src/preprocess/video.py`)
  - Audio extraction + MFCC features (`src/preprocess/audio.py`)
  - Transcription with Whisper (`src/preprocess/context.py`)
- ✅ Initial pipeline runner (`src/predict.py`).

## Next Steps
- Implement **feature extraction**:
  - Visual embeddings with pretrained ResNet/ViT.
  - Audio embeddings from spectrogram/MFCC.
  - Text embeddings from transcripts.
- Save extracted features for training.

## Future Goals
- Train unimodal models (visual, audio, text).
- Implement multi-modal fusion model.
- Evaluate performance on benchmark dataset.
- (Optional) Lightweight demo app/UI.

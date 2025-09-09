
# --- src/preprocess/audio.py ---
"""
Audio preprocessing utilities:
- extract_audio_with_ffmpeg: uses ffmpeg to produce a wav file
- extract_mfcc: uses librosa to compute MFCCs and save as numpy
"""

import subprocess
import numpy as np
import librosa


def extract_audio_with_ffmpeg(video_path: str, out_wav: str, sr: int = 16000):
    """Extract audio track from video using ffmpeg and resample to sr."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ar", str(sr), "-ac", "1", "-f", "wav", str(out_wav)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_wav


def extract_mfcc(wav_path: str, n_mfcc: int = 40, sr: int = 16000, hop_length: int = 512):
    """Load audio and return MFCC feature array (n_mfcc x time_frames)."""
    y, sr = librosa.load(wav_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    # optional: mean & var normalize per coefficient
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-9)
    return mfcc

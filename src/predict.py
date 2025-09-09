
# --- src/predict.py ---
"""
Minimal end-to-end runner that uses preprocessing utilities to extract features and
returns a dummy prediction. Replace model loading & fusion logic with real models later.
"""

import argparse
from pathlib import Path
from src.preprocess.video import extract_frames, detect_and_crop_faces
from src.preprocess.audio import extract_audio_with_ffmpeg, extract_mfcc
from src.preprocess.context import transcribe_with_whisper, simple_transcribe_placeholder


def run_pipeline(video_path: str, stmp_dir: str = 'data'):
    tmp_dir = Path(tmp_dir)
    frames_dir = tmp_dir / 'processed_frames'
    faces_dir = tmp_dir / 'faces'
    audio_wav = tmp_dir / 'extracted_audio' / (Path(video_path).stem + '.wav')
    transcript_txt = tmp_dir / 'transcripts' / (Path(video_path).stem + '.txt')

    frames = extract_frames(video_path, str(frames_dir), fps=1)
    print(f'Extracted {len(frames)} frames')

    all_crops = []
    for f in frames:
        crops = detect_and_crop_faces(f, str(faces_dir))
        all_crops.extend(crops)
    print(f'Extracted {len(all_crops)} face crops')

    audio_wav.parent.mkdir(parents=True, exist_ok=True)
    extract_audio_with_ffmpeg(video_path, str(audio_wav))
    print(f'Extracted audio to {audio_wav}')

    mfcc = extract_mfcc(str(audio_wav))
    print('Computed MFCC with shape', mfcc.shape)

    try:
        transcribe_with_whisper(video_path, str(transcript_txt))
        print('Transcript saved to', transcript_txt)
    except Exception:
        simple_transcribe_placeholder(video_path, str(transcript_txt))
        print('Used placeholder transcript')

    # Placeholder prediction logic
    # TODO: load visual/audio/context models and fuse their outputs
    prediction = {
        'video': 0.1,
        'audio': 0.05,
        'context': 0.0,
        'fused_score': 0.08,
        'label': 'real'
    }
    return prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--tmp', default='data', help='Temporary data dir')
    args = parser.parse_args()
    out = run_pipeline(args.video, args.tmp)
    print('Prediction: ', out)


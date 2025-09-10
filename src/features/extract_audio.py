import argparse
import os
import numpy as np
import librosa
from tqdm import tqdm


def extract_mfcc(audio_path, sr=16000, n_mfcc=40):
    """Load audio and extract MFCC features."""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)  # average over time → fixed-size vector
        return mfcc_mean
    except Exception as e:
        print(f"⚠️ Skipping {audio_path}: {e}")
        return None


def extract_features(input_dir, output_file, sr=16000, n_mfcc=40):
    """Extract MFCC features for all audio files in a folder."""
    features = {}
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.mp3', '.flac'))]

    for file in tqdm(files, desc="Processing audio"):
        path = os.path.join(input_dir, file)
        feat = extract_mfcc(path, sr=sr, n_mfcc=n_mfcc)
        if feat is not None:
            features[file] = feat

    # Save to .npy file
    np.save(output_file, features)
    print(f"✅ Saved {len(features)} audio features to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract audio features (MFCCs)")
    parser.add_argument("--input", required=True, help="Directory with audio files")
    parser.add_argument("--output", required=True, help="Output .npy file")
    args = parser.parse_args()

    extract_features(args.input, args.output)


if __name__ == "__main__":
    main()


# Example: extract MFCC features from audio
# python -m src.features.extract_audio --input data/audio --output data/features/audio.npy
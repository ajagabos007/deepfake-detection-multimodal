import argparse
import numpy as np
import os


def build_dataset(visual_file, audio_file, output_file):
    """
    Merge visual and audio features into a multimodal dataset.
    Assumes keys (filenames) match between visual.npy and audio.npy.
    """
    # Load features
    visual_features = np.load(visual_file, allow_pickle=True).item()
    audio_features = np.load(audio_file, allow_pickle=True).item()

    dataset = {}
    common_keys = set(visual_features.keys()) & set(audio_features.keys())

    print(f"🔗 Found {len(common_keys)} matching samples between visual & audio")

    for key in common_keys:
        dataset[key] = {
            "visual": visual_features[key],
            "audio": audio_features[key]
        }

    # Save merged dataset
    np.save(output_file, dataset)
    print(f"✅ Multimodal dataset saved to {output_file} ({len(dataset)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Build multimodal dataset from visual & audio features")
    parser.add_argument("--visual", required=True, help="Path to visual features .npy")
    parser.add_argument("--audio", required=True, help="Path to audio features .npy")
    parser.add_argument("--output", required=True, help="Output .npy dataset file")
    args = parser.parse_args()

    build_dataset(args.visual, args.audio, args.output)


if __name__ == "__main__":
    main()

# Merge extracted features into a multimodal dataset
# python -m src.features.dataset_builder \
#     --visual data/features/visual.npy \
#     --audio data/features/audio.npy \
#     --output data/features/multimodal.npy

# 📦 Output Structure
#
# The saved .npy dataset will look like:
# {
#   "video1.mp4": {
#       "visual": [2048-d feature vector],
#       "audio": [40-d MFCC vector]
#   },
#   "video2.mp4": {
#       "visual": [...],
#       "audio": [...]
#   }
# }

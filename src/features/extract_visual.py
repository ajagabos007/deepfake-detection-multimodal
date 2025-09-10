import argparse
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def load_model():
    """Load pretrained ResNet50 as feature extractor (no classifier)."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model = nn.Sequential(*list(model.children())[:-1])  # remove FC layer
    model.eval()
    return model


def build_transform():
    """Image preprocessing pipeline (resize, normalize)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225],   # ImageNet stds
        )
    ])


def extract_embeddings(input_dir, output_file, batch_size=16):
    """Extract visual embeddings for all images in a folder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model().to(device)
    transform = build_transform()

    embeddings = {}
    images, names = [], []

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for file in tqdm(files, desc="Processing images"):
        path = os.path.join(input_dir, file)
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = transform(img)
            images.append(img_tensor)
            names.append(file)
        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")

        # Batch process
        if len(images) >= batch_size:
            with torch.no_grad():
                batch = torch.stack(images).to(device)
                feats = model(batch).squeeze(-1).squeeze(-1)  # shape: [B, 2048]
                feats = feats.cpu().numpy()
                for n, f in zip(names, feats):
                    embeddings[n] = f
            images, names = [], []

    # Process remaining
    if images:
        with torch.no_grad():
            batch = torch.stack(images).to(device)
            feats = model(batch).squeeze(-1).squeeze(-1)
            feats = feats.cpu().numpy()
            for n, f in zip(names, feats):
                embeddings[n] = f

    # Save to .npy file
    np.save(output_file, embeddings)
    print(f"✅ Saved {len(embeddings)} embeddings to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract visual features using ResNet50")
    parser.add_argument("--input", required=True, help="Directory with face images")
    parser.add_argument("--output", required=True, help="Output .npy file")
    args = parser.parse_args()

    extract_embeddings(args.input, args.output)


if __name__ == "__main__":
    main()


# Example: extract embeddings from cropped faces
# python -m src.features.extract_visual --input data/faces --output data/features/visual.npy

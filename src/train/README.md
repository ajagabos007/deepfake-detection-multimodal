# Training Scripts

This folder contains training scripts for each modality (visual, audio, context) as well as the multimodal fusion model.

## 📂 Files

- `train_visual.py` → trains `VisualNet` on extracted frame-level features.
- `train_audio.py` → trains `AudioNet` on extracted audio (e.g., MFCC) features.
- `train_context.py` → trains `ContextNet` on textual/transcript features.
- `train_fusion.py` → combines the three modalities into `FusionNet`.

---

## 🛠️ Usage

Make sure you have pre-extracted features in `data/features/`:
- `visual.npy`
- `audio.npy`
- `context.npy`
- `labels.npy` (binary labels: `0 = real`, `1 = fake`)

Each script expects NumPy arrays and saves the trained model in `models/`.

### Example: Train visual model
```bash
python -m src.train.train_visual --epochs 10 --batch_size 32

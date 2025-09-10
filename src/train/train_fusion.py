# src/train/train_fusion.py
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
from src.models.fusion_net import FusionNet
from sklearn.metrics import accuracy_score, roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultimodalDataset(Dataset):
    """
    Expects a .npy file saved as a dict:
      key -> {"visual": np.array, "audio": np.array, "label": 0 or 1 (optional)}
    If label missing, we try to infer label from filename:
      - 'fake' in name -> 1
      - 'real' in name -> 0
    Samples without label and without those substrings are skipped.
    """

    def __init__(self, npy_file):
        data = np.load(npy_file, allow_pickle=True).item()
        self.keys = []
        self.visuals = []
        self.audios = []
        self.labels = []

        for k, v in data.items():
            # require both modalities
            if "visual" not in v or "audio" not in v:
                continue

            label = v.get("label", None)
            if label is None:
                name_lower = str(k).lower()
                if "fake" in name_lower:
                    label = 1
                elif "real" in name_lower:
                    label = 0
                else:
                    # skip ambiguous sample
                    # (you can extend this behavior by providing a label mapping file)
                    continue

            self.keys.append(k)
            self.visuals.append(np.asarray(v["visual"], dtype=np.float32))
            self.audios.append(np.asarray(v["audio"], dtype=np.float32))
            self.labels.append(int(label))

        if len(self.labels) == 0:
            raise RuntimeError("No labeled samples found in dataset. Provide labels or use filenames with 'fake'/'real'.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "visual": torch.from_numpy(self.visuals[idx]),
            "audio": torch.from_numpy(self.audios[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def collate_fn(batch):
    visuals = torch.stack([b["visual"] for b in batch])
    audios = torch.stack([b["audio"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return visuals, audios, labels


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []
    preds, trues = [], []
    for visuals, audios, labels in tqdm(loader, desc="train"):
        visuals = visuals.to(DEVICE)
        audios = audios.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(visuals, audios)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds.extend(probs.tolist())
        trues.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = float(np.mean(losses))
    try:
        auc = roc_auc_score(trues, preds)
    except Exception:
        auc = float("nan")
    return avg_loss, auc


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    losses = []
    preds, trues = [], []
    for visuals, audios, labels in tqdm(loader, desc="eval"):
        visuals = visuals.to(DEVICE)
        audios = audios.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(visuals, audios)
        loss = criterion(logits, labels)
        losses.append(loss.item())

        probs = torch.sigmoid(logits).cpu().numpy()
        preds.extend(probs.tolist())
        trues.extend(labels.cpu().numpy().tolist())

    avg_loss = float(np.mean(losses))
    try:
        auc = roc_auc_score(trues, preds)
        pred_labels = [1 if p >= 0.5 else 0 for p in preds]
        acc = accuracy_score(trues, pred_labels)
    except Exception:
        auc, acc = float("nan"), float("nan")
    return avg_loss, auc, acc


def main(args):
    ds = MultimodalDataset(args.dataset)
    n = len(ds)
    val_size = max(1, int(n * args.val_ratio))
    train_size = n - val_size
    train_set, val_set = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Infer dims from a sample
    sample = ds[0]
    visual_dim = sample["visual"].shape[0]
    audio_dim = sample["audio"].shape[0]
    print(f"Dataset: {n} samples | visual_dim={visual_dim} audio_dim={audio_dim}")

    model = FusionNet(visual_dim=visual_dim, audio_dim=audio_dim,
                      v_proj=args.v_proj, a_proj=args.a_proj,
                      hidden=args.hidden, dropout=args.dropout).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = lambda logits, labels: F.binary_cross_entropy_with_logits(logits, labels)

    best_auc = -1.0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_auc, val_acc = eval_epoch(model, val_loader, criterion)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_auc={train_auc:.4f} "
              f"| val_loss={val_loss:.4f} val_auc={val_auc:.4f} val_acc={val_acc:.4f}")

        # save best by val_auc
        if val_auc > best_auc:
            best_auc = val_auc
            ckpt_path = os.path.join(args.out_dir, f"fusion_best_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "visual_dim": visual_dim,
                "audio_dim": audio_dim
            }, ckpt_path)
            print(f"Saved best model to {ckpt_path} (val_auc={val_auc:.4f})")

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to multimodal .npy created by dataset_builder")
    parser.add_argument("--out_dir", default="models", help="Where to save checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--v_proj", type=int, default=512)
    parser.add_argument("--a_proj", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    main(args)

#     Quick Usage & Notes
#
# 1 Train (example):

# python -m src.train.train_fusion \
#   --dataset data/features/multimodal.npy \
#   --out_dir models \
#   --epochs 8 \
#   --batch_size 32


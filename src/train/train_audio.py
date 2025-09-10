# src/train/train_audio.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.audio_net import AudioNet

def train_audio(X_train, y_train, epochs=10, batch_size=32, lr=1e-3, device="cpu"):
    """
    Train AudioNet on extracted audio features.
    X_train: numpy array of audio features [N, 40]
    y_train: numpy array of labels [N]
    """
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AudioNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "models/audio_model.pth")
    print("✅ Audio model saved at models/audio_model.pth")
    return model

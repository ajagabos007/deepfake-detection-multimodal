# src/train/train_visual.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.visual_net import VisualNet

def train_visual(X_train, y_train, epochs=10, batch_size=32, lr=1e-3, device="cpu"):
    """
    Train VisualNet on extracted visual features.
    X_train: numpy array of visual features [N, 2048]
    y_train: numpy array of labels [N]
    """
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VisualNet().to(device)
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

    torch.save(model.state_dict(), "models/visual_model.pth")
    print("✅ Visual model saved at models/visual_model.pth")
    return model

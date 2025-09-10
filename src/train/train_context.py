# src/train/train_context.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.context_net import ContextNet

def train_context(X_train, y_train, epochs=10, batch_size=32, lr=1e-3, device="cpu"):
    """
    Train ContextNet on extracted transcript features.
    X_train: numpy array of context features [N, seq_len, emb_dim]
    y_train: numpy array of labels [N]
    """
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ContextNet().to(device)
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

    torch.save(model.state_dict(), "models/context_model.pth")
    print("✅ Context model saved at models/context_model.pth")
    return model

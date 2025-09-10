import torch
import torch.nn as nn

class ContextNet(nn.Module):
    """
    Text-based model for processing transcripts or subtitles.
    Simple baseline: Embedding + LSTM -> feature vector
    Future extension: Replace LSTM with Transformer Encoder.
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, feature_dim=128, num_layers=1):
        super(ContextNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # LSTM baseline
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # Projection to feature vector
        self.fc = nn.Linear(hidden_dim, feature_dim)

        # --- Future extension: Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.use_transformer = False  # Toggle later

    def forward(self, x):
        """
        x: (batch, seq_len) token ids
        """
        x = self.embedding(x)  # (batch, seq_len, embed_dim)

        if self.use_transformer:
            # Transformer expects (seq_len, batch, embed_dim)
            x = x.transpose(0, 1)
            x = self.transformer_encoder(x)
            x = x.mean(dim=0)  # Mean pooling
        else:
            # LSTM baseline
            _, (h, _) = self.lstm(x)
            x = h[-1]  # Last hidden state

        x = self.fc(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionNet(nn.Module):
    """
    Late-fusion MLP baseline:
    - Visual projector: 2048 -> 512
    - Audio projector:   40  -> 128
    - Context projector: 300 -> 128 (optional, future step)
    - Concatenate -> MLP -> Logit
    """

    def __init__(self,
                 visual_dim=2048,
                 audio_dim=40,
                 context_dim=None,  # None for now
                 v_proj=512,
                 a_proj=128,
                 c_proj=128,
                 hidden=256,
                 dropout=0.3):
        super().__init__()

        # visual branch
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, v_proj),
            nn.BatchNorm1d(v_proj),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # audio branch
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, a_proj),
            nn.BatchNorm1d(a_proj),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
        )

        # context branch (future)
        if context_dim:
            self.context_proj = nn.Sequential(
                nn.Linear(context_dim, c_proj),
                nn.BatchNorm1d(c_proj),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout / 2),
            )
        else:
            self.context_proj = None

        # fusion MLP
        fused_dim = v_proj + a_proj + (c_proj if context_dim else 0)
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden // 2, 1)  # binary classification logit
        )

    def forward(self, visual, audio, context=None):
        v = self.visual_proj(visual)
        a = self.audio_proj(audio)
        feats = [v, a]

        if self.context_proj and context is not None:
            c = self.context_proj(context)
            feats.append(c)

        x = torch.cat(feats, dim=1)
        logit = self.mlp(x)
        return logit.squeeze(1)

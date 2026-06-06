"""
Dilated 1-D temporal convolutional network for bounce detection.

Operates on the shared TCN channel stack (TCN_CHANNELS, see train/config.py): a
(7, T) per-frame trajectory-feature sequence per clip. A stack of dilated
residual conv blocks gives each output frame a wide receptive field, so the net
sees the velocity reversal / acceleration spike surrounding a bounce. The output
is a per-frame bounce logit; sigmoid + the shared decode/match turns it into
events downstream.

Kept small (~50-150k params) because positives are scarce (~473 bounce frames).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import channel_stack
from train.config import TCN, TCN_CHANNELS


# ─── model ─────────────────────────────────────────────────────────────────────
class _ResidualBlock(nn.Module):
    """Dilated 'same'-padded Conv1d → ReLU → Dropout, with a residual add. The
    first block uses a 1x1 conv to lift in_ch→hidden so the skip connection
    matches channels."""

    def __init__(self, in_ch, hidden, kernel, dilation, dropout):
        super().__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_ch, hidden, kernel, padding="same", dilation=dilation))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.down = nn.Conv1d(in_ch, hidden, 1) if in_ch != hidden else None

    def forward(self, x):
        res = x if self.down is None else self.down(x)
        out = self.dropout(self.relu(self.conv(x)))
        return out + res


class BounceTCN(nn.Module):
    def __init__(self, in_ch=7, hidden=32, levels=4, kernel=3, dropout=0.1):
        super().__init__()
        blocks = []
        ch = in_ch
        for l in range(levels):
            blocks.append(_ResidualBlock(ch, hidden, kernel, 2 ** l, dropout))
            ch = hidden
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv1d(hidden, 1, 1)

    def forward(self, x):                # x: (B, in_ch, T)
        h = self.blocks(x)
        return self.head(h).squeeze(1)   # (B, T)


# ─── shared SCORER CONTRACT ────────────────────────────────────────────────────
class Scorer:
    def __init__(self, checkpoint_path, device="cpu"):
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model = BounceTCN(**ckpt["cfg"]).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.threshold = ckpt.get("threshold")
        self.device = device

    @torch.no_grad()
    def score(self, feats, names, traj):
        cs = channel_stack(feats, TCN_CHANNELS)        # (C, T)
        C, T = cs.shape
        window, stride = TCN["window"], TCN["stride"]

        if T <= window:                                 # whole clip in one window
            pad = window - T
            win = np.pad(cs, ((0, 0), (0, pad)), mode="edge")
            x = torch.from_numpy(win).unsqueeze(0).to(self.device)
            probs = torch.sigmoid(self.model(x))[0].cpu().numpy()
            return probs[:T].astype(np.float32)         # DON'T zero invalid frames

        starts = list(range(0, T - window + 1, stride))
        if starts[-1] + window < T:                     # cover the tail
            starts.append(T - window)

        acc = np.zeros(T, dtype=np.float64)
        cnt = np.zeros(T, dtype=np.float64)
        for s in starts:                                # overlap-average back to (T,)
            e = s + window
            x = torch.from_numpy(cs[:, s:e]).unsqueeze(0).to(self.device)
            probs = torch.sigmoid(self.model(x))[0].cpu().numpy()
            acc[s:e] += probs
            cnt[s:e] += 1.0
        return (acc / np.maximum(cnt, 1.0)).astype(np.float32)

"""
S-KeepTrack: Simplified tracking-by-detection model with candidate association.

Architecture summary:
  1. Shared ResNet backbone extracts features for each frame.
  2. Target Classifier head produces score maps and candidate bounding boxes.
  3. Candidate Association Network matches candidates across consecutive frames.

We implement a lightweight version suitable for training from scratch on the
TrackNet dataset (no need for a full COCO-pretrained detection pipeline).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class ResNetBackbone(nn.Module):
    """ResNet-18 feature extractor outputting stride-16 feature maps."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1   # stride 4,  64-ch
        self.layer2 = base.layer2   # stride 8, 128-ch
        self.layer3 = base.layer3   # stride 16, 256-ch

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x   # (B, 256, H/16, W/16)


# ---------------------------------------------------------------------------
# Target Classifier / Score Map head
# ---------------------------------------------------------------------------

class TargetClassifier(nn.Module):
    """Produces a spatial score map from backbone features."""

    def __init__(self, in_ch: int = 256, mid_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, 1),
        )

    def forward(self, feat):
        return torch.sigmoid(self.net(feat))   # (B, 1, H', W')


# ---------------------------------------------------------------------------
# Candidate extraction: top-k positions from score map
# ---------------------------------------------------------------------------

def extract_candidates(score_map: torch.Tensor, feat: torch.Tensor,
                        k: int = 8):
    """
    score_map : (B, 1, H, W)
    feat       : (B, C, H, W)
    Returns:
        coords  : (B, k, 2)  — normalised (cx, cy) in [0, 1]
        scores  : (B, k)
        feat_k  : (B, k, C)
    """
    B, _, H, W = score_map.shape
    C = feat.shape[1]
    flat = score_map.view(B, -1)
    scores, indices = flat.topk(k, dim=1)

    ys = (indices // W).float() / H
    xs = (indices % W).float() / W
    coords = torch.stack([xs, ys], dim=2)   # (B, k, 2)

    feat_flat = feat.view(B, C, -1).permute(0, 2, 1)   # (B, H*W, C)
    feat_k = torch.gather(feat_flat, 1,
                          indices.unsqueeze(-1).expand(-1, -1, C))  # (B, k, C)

    return coords, scores, feat_k


# ---------------------------------------------------------------------------
# Candidate Association Network
# ---------------------------------------------------------------------------

class CandidateEmbedder(nn.Module):
    """Embeds (coord, score, feature) tuples into a fixed-dim vector."""

    def __init__(self, feat_ch: int = 256, embed_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_ch + 3, embed_dim),   # +3 for (cx, cy, score)
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, coords, scores, feats):
        """
        coords : (B, k, 2)
        scores : (B, k)
        feats  : (B, k, C)
        """
        inp = torch.cat([feats, coords, scores.unsqueeze(-1)], dim=-1)
        return self.mlp(inp)   # (B, k, embed_dim)


class AssociationHead(nn.Module):
    """
    Computes pairwise association scores between candidates in two frames.
    Returns a (B, k, k) matrix of match probabilities.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(embed_dim * 2, 1)

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor):
        """
        emb1, emb2 : (B, k, D)
        Returns     : (B, k, k)  — sigmoid association scores
        """
        k = emb1.shape[1]
        e1 = emb1.unsqueeze(2).expand(-1, -1, k, -1)   # (B, k, k, D)
        e2 = emb2.unsqueeze(1).expand(-1, k, -1, -1)   # (B, k, k, D)
        pair = torch.cat([e1, e2], dim=-1)              # (B, k, k, 2D)
        return torch.sigmoid(self.proj(pair).squeeze(-1))  # (B, k, k)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class SKeepTrack(nn.Module):
    """
    Input : two consecutive frames (B, 3, H, W) each
    Output:
        score_map1, score_map2 : (B, 1, H', W') — for classification loss
        assoc_matrix           : (B, k, k)       — for association loss
        pred_coords            : (B, 2)           — predicted ball position
                                                    in frame2 (normalised)
    """

    def __init__(self, k: int = 8, pretrained: bool = True):
        super().__init__()
        self.k = k
        self.backbone = ResNetBackbone(pretrained=pretrained)
        self.classifier = TargetClassifier(in_ch=256, mid_ch=128)
        self.embedder = CandidateEmbedder(feat_ch=256, embed_dim=128)
        self.assoc = AssociationHead(embed_dim=128)

    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor):
        feat1 = self.backbone(frame1)
        feat2 = self.backbone(frame2)

        sm1 = self.classifier(feat1)
        sm2 = self.classifier(feat2)

        coords1, scores1, feats1 = extract_candidates(sm1, feat1, self.k)
        coords2, scores2, feats2 = extract_candidates(sm2, feat2, self.k)

        emb1 = self.embedder(coords1, scores1, feats1)
        emb2 = self.embedder(coords2, scores2, feats2)

        assoc = self.assoc(emb1, emb2)   # (B, k, k)

        # Predicted position: candidate in frame2 with highest association sum
        assoc_sum = assoc.sum(dim=1)                    # (B, k)
        best_idx = assoc_sum.argmax(dim=1)              # (B,)
        pred_coords = coords2[torch.arange(coords2.shape[0]), best_idx]  # (B, 2)

        return sm1, sm2, assoc, pred_coords

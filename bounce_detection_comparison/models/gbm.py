"""
Gradient-boosted per-frame bounce regressor.

CatBoost is preferred; if it isn't installed we fall back to scikit-learn's
HistGradientBoostingRegressor (mirroring the import-or-fallback pattern in
../tennis_ball_tracking_comparison/train/train_yolo11m.py). Either way the model
regresses the shared soft (Gaussian-in-time) bounce target from the 71 trajectory
features, and the per-frame score is decoded into events downstream.
"""

import numpy as np

from data.dataset import feature_matrix
from train.config import GBM

try:
    import joblib as _serializer
except ImportError:
    import pickle as _serializer


# ─── model factory (CatBoost preferred, sklearn fallback) ─────────────────────
def build_model(cfg=GBM):
    """Return a regressor: CatBoost if importable, else sklearn HistGBR."""
    try:
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=cfg["iterations"],
            learning_rate=cfg["learning_rate"],
            depth=cfg["depth"],
            loss_function="RMSE",
            verbose=False,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=cfg["iterations"],
            learning_rate=cfg["learning_rate"],
        )


# ─── persistence (single .pkl envelope) ───────────────────────────────────────
def save_model(model, threshold, backend, path):
    """Persist {"backend", "model", "threshold"} to a single .pkl."""
    env = {"backend": backend, "model": model, "threshold": threshold}
    if _serializer.__name__ == "joblib":
        _serializer.dump(env, path)
    else:
        with open(path, "wb") as f:
            _serializer.dump(env, f)


def load_model(path):
    if _serializer.__name__ == "joblib":
        return _serializer.load(path)
    with open(path, "rb") as f:
        return _serializer.load(f)


# ─── shared SCORER CONTRACT ────────────────────────────────────────────────────
class Scorer:
    def __init__(self, checkpoint_path, device="cpu"):
        env = load_model(checkpoint_path)
        self.model = env["model"]
        self.threshold = env.get("threshold")

    def score(self, feats, names, traj):
        X = feature_matrix(feats, names)          # (T, F)
        pred = self.model.predict(X)
        return np.clip(pred, 0.0, 1.0)            # (T,) in [0,1]; DON'T zero invalid frames

"""
Gradient-boosted per-frame bounce regressor (shared across boosting arms).

`build_model` constructs one of four interchangeable backends — scikit-learn
HistGradientBoosting (the `gbm` arm), XGBoost, LightGBM, or CatBoost. Every
backend regresses the same shared soft (Gaussian-in-time) bounce target from the
71 trajectory features, and the per-frame score is decoded into events
downstream. Because the `Scorer` only relies on `model.predict`, a single Scorer
serves all four arms; only training differs (each library's native fit / early
stopping, see train/train_gbm.py).
"""

import numpy as np

from data.dataset import feature_matrix

try:
    import joblib as _serializer
except ImportError:
    import pickle as _serializer


# ─── model factory (one regressor per boosting backend) ───────────────────────
def build_model(backend, cfg):
    """Return an unfitted regressor for the requested boosting backend."""
    if backend in ("histgbm", "gbm", "sklearn"):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=cfg["iterations"],
            learning_rate=cfg["learning_rate"],
        )
    if backend == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=cfg["iterations"],
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["depth"],
            objective="reg:squarederror",
            early_stopping_rounds=cfg["early_stopping_rounds"],
            n_jobs=-1,
            verbosity=0,
        )
    if backend == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=cfg["iterations"],
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["depth"],
            num_leaves=cfg["num_leaves"],
            objective="regression",
            n_jobs=-1,
            verbose=-1,
        )
    if backend == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=cfg["iterations"],
            learning_rate=cfg["learning_rate"],
            depth=cfg["depth"],
            loss_function="RMSE",
            verbose=False,
        )
    raise ValueError(f"unknown boosting backend: {backend}")


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

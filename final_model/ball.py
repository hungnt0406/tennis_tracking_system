"""Ball tracking inference wrapper around the TrackNetV4 model."""
import numpy as np
import torch

from ._loader import import_project, BALL_ROOT

_mods = import_project(
    BALL_ROOT,
    ["models.tracknetv4", "models.tracknet", "data.preprocessing"],
    "_fm_ball",
)
TrackNetV4 = _mods["models.tracknetv4"].TrackNetV4
intensity_to_coords = _mods["models.tracknet"].intensity_to_coords
resize_frame = _mods["data.preprocessing"].resize_frame
normalize = _mods["data.preprocessing"].normalize


def _clamp(i, n):
    return max(0, min(i, n - 1))


class BallTracker:
    def __init__(self, checkpoint: str, device: str = "cpu", decode: str = "hough"):
        self.device = device
        self.use_hough = decode == "hough"
        self.model = TrackNetV4().to(device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        self.model.eval()

    def infer_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        T = len(frames)
        chans = [normalize(resize_frame(f)) for f in frames]  # each (3, 368, 640)
        triplets = [
            np.concatenate([chans[_clamp(i - 1, T)], chans[i], chans[_clamp(i + 1, T)]])
            for i in range(T)
        ]  # each (9, 368, 640)

        coords = []
        with torch.no_grad():
            for s in range(0, T, 8):
                batch = np.stack(triplets[s:s + 8])
                inp = torch.from_numpy(batch).to(self.device)
                logits = self.model(inp)
                intensity = logits.argmax(dim=1).cpu()
                coords.append(intensity_to_coords(intensity, use_hough=self.use_hough).numpy())
        return np.concatenate(coords).astype(np.float32)

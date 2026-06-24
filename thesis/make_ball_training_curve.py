#!/usr/bin/env python3
"""Generate the TrackNetV4 full-dataset training-curve figure for Chapter 4.

Parses the beautified epoch summaries in the from-scratch training log and writes
a single twin-axis PNG (loss on a log scale, val acc@5px on a linear scale) into
thesis/img/. No training is run; all numbers come from the log.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path("/Users/hungcucu/Documents/usth/tennis_tracking_system")
SRC_LOG = REPO / "remote_logs/ict17/tracknetv4_fromscratch_train.log"
OUT_DIR = REPO / "thesis/img"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style constants (kept consistent with make_court_figures.py)
# ---------------------------------------------------------------------------
C_TRAIN = "#5b7c99"  # muted slate blue
C_ACC = "#2ca02c"    # emphasis green (matches the selected-model colour)

TITLE_FS = 13
LABEL_FS = 11
TICK_FS = 10

EPOCH_RE = re.compile(
    r"^Epoch\s+(\d+)\s+\|\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)\s+val_acc@5px=([\d.]+)"
)


def parse_log():
    epochs, train_loss, val_loss, val_acc = [], [], [], []
    for line in SRC_LOG.read_text().splitlines():
        m = EPOCH_RE.match(line)
        if m:
            epochs.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
            val_loss.append(float(m.group(3)))
            val_acc.append(float(m.group(4)))
    return epochs, train_loss, val_loss, val_acc


def make_figure():
    epochs, train_loss, _val_loss, val_acc = parse_log()
    best_i = max(range(len(val_acc)), key=lambda i: val_acc[i])
    best_epoch, best_acc = epochs[best_i], val_acc[best_i]

    fig, ax_loss = plt.subplots(figsize=(6.4, 3.8))

    # Left axis: train loss on a linear scale.
    l1, = ax_loss.plot(epochs, train_loss, color=C_TRAIN, linewidth=1.8,
                       label="train loss")
    ax_loss.set_xlabel("Epoch", fontsize=LABEL_FS)
    ax_loss.set_ylabel("Loss", fontsize=LABEL_FS)
    ax_loss.set_xlim(1, epochs[-1])
    ax_loss.set_ylim(bottom=0)
    ax_loss.grid(alpha=0.3, which="both")
    ax_loss.tick_params(labelsize=TICK_FS)

    # Right axis: validation accuracy at 5px on a linear scale.
    ax_acc = ax_loss.twinx()
    l3, = ax_acc.plot(epochs, val_acc, color=C_ACC, linewidth=2.0,
                      label="val acc@5px")
    ax_acc.set_ylabel("Validation acc@5px", fontsize=LABEL_FS, color=C_ACC)
    ax_acc.set_ylim(0, 1.0)
    ax_acc.tick_params(axis="y", labelsize=TICK_FS, labelcolor=C_ACC)

    # Mark the best validation checkpoint.
    m_best = ax_acc.scatter([best_epoch], [best_acc], color=C_ACC, s=45,
                            zorder=5, edgecolors="black", linewidths=0.8,
                            label=f"best (ep {best_epoch}, {best_acc:.3f})")
    ax_acc.axvline(best_epoch, color=C_ACC, linestyle="--", linewidth=0.9,
                   alpha=0.5)

    ax_loss.set_title("TrackNetV4 full-dataset training curve", fontsize=TITLE_FS)
    handles = [l1, l3, m_best]
    ax_loss.legend(handles, [h.get_label() for h in handles],
                   fontsize=9, loc="center right")

    fig.tight_layout()
    out = OUT_DIR / "ball_v4_training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}  ({len(epochs)} epochs, best val acc@5px={best_acc:.3f} "
          f"at epoch {best_epoch})")


if __name__ == "__main__":
    make_figure()

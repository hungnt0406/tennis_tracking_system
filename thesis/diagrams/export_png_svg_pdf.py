#!/usr/bin/env python3
"""Render the TrackNetV4 figure to PNG/SVG/PDF. Each layer is drawn as a set of
thin overlaid sheets receding to the back (up-right), not a solid block."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, Circle, FancyArrowPatch
import matplotlib.image as mpimg

DIAG = "/Users/hungcucu/Documents/usth/tennis_tracking_system/thesis/diagrams"
HT = 600
fig, ax = plt.subplots(figsize=(16.4, 6), dpi=200)
ax.set_xlim(0, 1640); ax.set_ylim(0, HT); ax.axis("off")
def Y(y): return HT - y

GRAY_L = "#DCDCDC"; BOT_L = "#C9C9C9"; FEAT_F = "#AEC9E8"; ATT_F = "#3A63E0"
MBLUE = "#2F5BEA"; RED = "#D6463B"; BLACK = "#1A1A1A"
DD = (0, (6, 3, 1, 3))
KIND = {"g": GRAY_L, "b": BOT_L, "feat": FEAT_F, "att": ATT_F}

# ---- thin layered-sheet projection ----
SHW, LEAN, KSH, SDPX, SDPY = 8, 4, 4, 7, 6     # sheet width, lean, #sheets, depth/sheet

def _rgb(c): c = c.lstrip("#"); return tuple(int(c[i:i+2], 16) for i in (0, 2, 4))
def shade(c, f):
    r, g, b = _rgb(c)
    if f >= 0: r, g, b = (int(v + (255 - v) * f) for v in (r, g, b))
    else: r, g, b = (int(v * (1 + f)) for v in (r, g, b))
    return "#%02X%02X%02X" % (r, g, b)
def poly(pts, fill, stroke, lw=0.6, z=3):
    ax.add_patch(Polygon([(px, Y(py)) for px, py in pts], closed=True, facecolor=fill,
                         edgecolor=stroke, linewidth=lw, joinstyle="round", zorder=z))

def topA(left, h, cy):  return (left + SHW / 2 + (KSH - 1) * SDPX / 2, cy - h / 2 - (KSH - 1) * SDPY)
def botA(left, h, cy):  return (left + SHW / 2, cy + h / 2)

def layer(left, h, kind, cy, z=3):
    base = KIND[kind]; st = shade(base, -0.45)
    for k in range(KSH - 1, -1, -1):            # back (k=KSH-1) first
        ox, oy = k * SDPX, -k * SDPY
        fk = shade(base, 0.11 * k)              # back sheets lighter -> recede
        yb, yt = cy + h / 2 + oy, cy - h / 2 + oy
        x = left + ox
        poly([(x, yb), (x + SHW, yb), (x + SHW + LEAN, yt), (x + LEAN, yt)],
             fk, st, z=z + (KSH - k) * 0.05)

def image_cell(x, y, w, h, path):
    ax.imshow(mpimg.imread(path), extent=[x, x + w, Y(y + h), Y(y)], zorder=5, interpolation="bilinear")
def rrect(x, y, w, h, stroke, fill="none", dash=False, lw=2):
    ax.add_patch(FancyBboxPatch((x, Y(y + h)), w, h, boxstyle="round,pad=0,rounding_size=11",
                 facecolor=("none" if fill == "none" else fill), edgecolor=stroke, linewidth=lw,
                 linestyle=(DD if dash else "-"), zorder=2))
def text(x, y, w, h, s, size=12, bold=True, color=BLACK, align="left"):
    ax.text(x, Y(y + h / 2), s, fontsize=size, fontweight="bold" if bold else "normal",
            color=color, ha=align, va="center", family="DejaVu Sans", zorder=10)
def edge(pts, stroke=BLACK, lw=1.7, dash=False, arrow=True, z=6):
    P = [(px, Y(py)) for px, py in pts]
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]
        if i == len(P) - 2 and arrow:
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=stroke,
                         lw=lw, linestyle=(DD if dash else "-"), shrinkA=0, shrinkB=0, zorder=z))
        else:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=stroke, lw=lw,
                    linestyle=(DD if dash else "-"), zorder=z, solid_capstyle="round")
def line(pts, stroke=BLACK, lw=1.2, z=6): edge(pts, stroke=stroke, lw=lw, arrow=False, z=z)
def odot(cx, cy, r=12, stroke=MBLUE):
    ax.add_patch(Circle((cx, Y(cy)), r, facecolor="white", edgecolor=stroke, lw=2, zorder=11))
    ax.add_patch(Circle((cx, Y(cy)), 3.2, facecolor=stroke, edgecolor=stroke, zorder=12))
def ring(cx, cy, r=8, stroke=RED, lw=2.4):
    ax.add_patch(Circle((cx, Y(cy)), r, facecolor="none", edgecolor=stroke, lw=lw, zorder=13))

# ===========================================================================
CY, X0, PITCH = 290, 250, 34
BOX_X, BOX_Y, BOX_W, BOX_H = 205, 156, 930, 256
rrect(BOX_X, BOX_Y, BOX_W, BOX_H, BLACK, lw=2.2)

seq = []
def add(h, kind, stage, n=1):
    for _ in range(n): seq.append([h, kind, stage])
add(170, "g", 0, 2); add(140, "g", 1, 2); add(110, "g", 2, 3); add(82, "g", 3, 3)
add(60, "b", 4, 3); add(82, "g", 5, 3); add(110, "g", 6, 3)
add(140, "g", 7); add(140, "feat", 7); add(130, "att", None)
add(170, "g", 8); add(170, "feat", 8); add(160, "att", None)

slabs = []
for i, (h, kind, stage) in enumerate(seq):
    slabs.append(dict(left=X0 + PITCH * i, h=h, kind=kind, stage=stage, i=i))
N = len(seq)
for sl in slabs:
    layer(sl["left"], sl["h"], sl["kind"], CY, z=3 + (N - sl["i"]) * 0.4)

def by_stage(s): return [x for x in slabs if x["stage"] == s]
for es, ds, ay in [(0, 8, 178), (1, 7, 190), (2, 6, 202), (3, 5, 214)]:
    a = by_stage(es)[-1]; b = by_stage(ds)[0]
    pa = topA(a["left"], a["h"], CY); pb = topA(b["left"], b["h"], CY)
    edge([pa, (pa[0], ay), (pb[0], ay), pb], stroke="#9AA0A6", lw=1.3)

IMG_W, IMG_H = 165, 98
IX, IY = 42, CY - IMG_H // 2
image_cell(IX, IY, IMG_W, IMG_H, f"{DIAG}/_input_frames.png")
edge([(IX + IMG_W + 4, CY), (X0 + 2, CY)], lw=2.0)

MX, MY, MW, MH = 330, 50, 360, 104
rrect(MX, MY, MW, MH, MBLUE, dash=True, lw=2.2); mc = MY + MH // 2
line([(MX + 24, mc + 12), (MX + MW - 24, mc + 12)], stroke="#3A3A3A", lw=1.1)
line([(MX + MW // 2, mc + 12), (MX + MW // 2, mc - 18)], stroke="#3A3A3A", lw=1.0)
cxm = MX + MW // 2
edge([(MX + 20, mc - 2), (cxm - 40, mc - 2), (cxm - 6, mc + 26), (cxm + 4, mc - 26),
      (cxm + 14, mc - 2), (MX + MW - 20, mc - 2)], stroke=RED, lw=2.2, arrow=False)
edge([(IX + 22, IY + 6), (IX + 22, mc), (MX + 6, mc)], stroke=MBLUE, lw=2.0)

# attention maps (top) as a layered-sheet group
AX, AH, AYC = 770, 72, 104
edge([(MX + MW, mc), (AX - 4, AYC)], stroke=MBLUE, lw=2.0)
layer(AX, AH, "att", AYC, z=40)
ring(AX + 4, AYC, r=5)

atts = [s for s in slabs if s["kind"] == "att"]; feats = [s for s in slabs if s["kind"] == "feat"]
a_half, a_full = atts; f_half, f_full = feats
pf = topA(a_full["left"], a_full["h"], CY); ph = topA(a_half["left"], a_half["h"], CY)
edge([(AX + 52, AYC), (pf[0], AYC), pf], stroke=MBLUE, lw=2.0, dash=True)
edge([(ph[0], AYC), ph], stroke=MBLUE, lw=2.0, dash=True)
for f, a in [(f_half, a_half), (f_full, a_full)]:
    bf = botA(f["left"], f["h"], CY); ba = botA(a["left"], a["h"], CY)
    ox = (bf[0] + ba[0]) / 2; oy = CY + max(f["h"], a["h"]) / 2 + 24
    odot(ox, oy)
    line([bf, (bf[0], oy), (ox - 8, oy - 9)], stroke=MBLUE, lw=1.4)
    line([ba, (ba[0], oy), (ox + 8, oy - 9)], stroke=MBLUE, lw=1.4)

edge([(BOX_X + BOX_W, CY), (BOX_X + BOX_W + 60, CY)], lw=2.0)
OX, OY = BOX_X + BOX_W + 62, CY - IMG_H // 2
image_cell(OX, OY, IMG_W, IMG_H, f"{DIAG}/_heatmap.png")

# ---- legend ----
LY = 452
def leg(x, s): text(x, LY - 8, 128, 46, s, size=11, bold=True)
groups = [30, 230, 430, 630, 830, 1030, 1230, 1430]
gx = groups[0]; image_cell(gx, LY, 40, 26, f"{DIAG}/_input_frames.png"); leg(gx + 48, "Input frames")
gx = groups[1]; rrect(gx, LY + 4, 48, 30, BLACK, lw=2); leg(gx + 58, "TrackNet\nbackbone")
gx = groups[2]; rrect(gx, LY + 4, 50, 30, MBLUE, dash=True, lw=2)
edge([(gx + 8, LY + 19), (gx + 21, LY + 28), (gx + 27, LY + 9), (gx + 42, LY + 19)], stroke=RED, lw=1.8, arrow=False); leg(gx + 62, "Motion attention\nmodule")
gx = groups[3]; edge([(gx + 16, LY + 2), (gx + 16, LY + 34)], stroke=MBLUE, lw=2.4, dash=True); leg(gx + 30, "Motion-aware\nfusion path")
gx = groups[4]; layer(gx, 30, "att", LY + 18, z=20); leg(gx + 52, "Attention maps")
gx = groups[5]; layer(gx, 30, "feat", LY + 18, z=20); leg(gx + 52, "Feature maps")
gx = groups[6]; odot(gx + 13, LY + 19, r=12); leg(gx + 34, "Element-wise\nmultiplication")
gx = groups[7]; image_cell(gx, LY, 40, 26, f"{DIAG}/_heatmap.png"); leg(gx + 48, "Predicted\nheatmaps")

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
for ext in ("png", "svg", "pdf"):
    plt.savefig(f"{DIAG}/tracknetv4_architecture.{ext}", bbox_inches="tight", pad_inches=0.1,
                transparent=(ext != "png"), facecolor="white" if ext == "png" else "none")
print("wrote png/svg/pdf to", DIAG)

#!/usr/bin/env python3
"""Build the two embedded raster assets for the TrackNetV4 figure:
  _input_frames.png : 3 real court frames, sheared + stacked, red ring on ball
  _heatmap.png      : 3 black frames, sheared + stacked, warm gaussian blob
Shear matches the slab projection (top edge shifted right by LEAN_K*height)."""
import os, math
from PIL import Image, ImageDraw

ROOT = "/Users/hungcucu/Documents/usth/tennis_tracking_system"
CLIP = f"{ROOT}/Dataset/game1/Clip1"
OUT  = f"{ROOT}/thesis/diagrams"
os.makedirs(OUT, exist_ok=True)

LEAN_K = 0.07
FH = 92                                   # display height of one frame
FW = round(FH * 1280 / 720)               # keep 16:9
SOFF = max(3, round(LEAN_K * FH))         # shear offset (top shifted right)
SDX, SDY, N = 11, 11, 3                   # stack offsets

frames = [("0000.jpg", 599, 423), ("0001.jpg", 601, 406), ("0002.jpg", 601, 388)]

def shear(img):
    """horizontal shear: top row shifted right by SOFF, bottom unchanged."""
    w, h = img.size
    return img.transform((w + SOFF, h), Image.AFFINE,
                         (1, SOFF / h, -SOFF, 0, 1, 0),
                         resample=Image.BICUBIC)

def ball_xy_sheared(bx, by):
    """ball pixel in the sheared frame coords."""
    sx = bx + SOFF * (1 - by / FH)
    return sx, by

def composite(make_frame):
    CW = FW + SOFF + (N - 1) * SDX
    CH = FH + (N - 1) * SDY
    canvas = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
    for k in range(N - 1, -1, -1):                # back (k=N-1) first
        fr = make_frame(k)                        # RGBA, size (FW+SOFF, FH)
        ox = k * SDX
        oy = (N - 1 - k) * SDY
        canvas.alpha_composite(fr, (ox, oy))
    return canvas

# ---- input frames -----------------------------------------------------------
def input_frame(k):
    name, bx, by = frames[k]
    im = Image.open(f"{CLIP}/{name}").convert("RGBA").resize((FW, FH), Image.LANCZOS)
    im = shear(im)
    d = ImageDraw.Draw(im)
    sbx, sby = ball_xy_sheared(bx * FW / 1280, by * FH / 720)
    r = 7
    d.ellipse([sbx - r, sby - r, sbx + r, sby + r], outline=(230, 60, 50, 255), width=3)
    return im

inp = composite(input_frame)
inp.save(f"{OUT}/_input_frames.png")

# ---- predicted heatmap ------------------------------------------------------
def heat_frame(k):
    name, bx, by = frames[k]
    base = Image.new("RGBA", (FW, FH), (8, 8, 10, 255))
    px = base.load()
    cx, cy = bx * FW / 1280, by * FH / 720
    sig = 6.0
    for y in range(FH):
        for x in range(FW):
            d2 = (x - cx) ** 2 + (y - cy) ** 2
            g = math.exp(-d2 / (2 * sig * sig))
            if g > 0.01:
                # black -> red -> orange -> white warm ramp
                r = min(255, int(40 + 215 * min(1, g * 1.4)))
                gg = min(255, int(g ** 1.6 * 230))
                b = min(255, int(g ** 3 * 200))
                px[x, y] = (max(r, px[x, y][0]), max(gg, px[x, y][1]),
                            max(b, px[x, y][2]), 255)
    im = shear(base)
    return im

hm = composite(heat_frame)
hm.save(f"{OUT}/_heatmap.png")

print("input_frames", inp.size)
print("heatmap", hm.size)
print("FW,FH,SOFF", FW, FH, SOFF)

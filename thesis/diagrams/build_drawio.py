#!/usr/bin/env python3
"""Generate the TrackNetV4 architecture figure as a draw.io (mxGraph XML) file.
Each layer = thin overlaid sheets (native leaning parallelograms) receding to
the back (up-right), not a solid block. Faithful to models/tracknetv4.py."""
import base64, xml.dom.minidom as MD

DIAG = "/Users/hungcucu/Documents/usth/tennis_tracking_system/thesis/diagrams"
def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
IMG_INPUT = b64(f"{DIAG}/_input_frames.png")
IMG_HEAT = b64(f"{DIAG}/_heatmap.png")

cells = []
_id = [10]
def nid():
    _id[0] += 1
    return f"n{_id[0]}"

GRAY_L = "#DCDCDC"; BOT_L = "#C9C9C9"; FEAT_F = "#AEC9E8"; ATT_F = "#3A63E0"
MBLUE = "#2F5BEA"; RED = "#D6463B"; BLACK = "#1A1A1A"
DASHDOT = "dashed=1;dashPattern=8 4 1 4;"
KIND = {"g": GRAY_L, "b": BOT_L, "feat": FEAT_F, "att": ATT_F}
SHW, LEAN, KSH, SDPX, SDPY = 8, 4, 4, 7, 6      # sheet width, lean, #sheets, depth/sheet

def _rgb(c): c = c.lstrip("#"); return tuple(int(c[i:i+2], 16) for i in (0, 2, 4))
def shade(c, f):
    r, g, b = _rgb(c)
    if f >= 0: r, g, b = (int(v + (255 - v) * f) for v in (r, g, b))
    else: r, g, b = (int(v * (1 + f)) for v in (r, g, b))
    return "#%02X%02X%02X" % (r, g, b)

def topA(left, h, cy): return (left + SHW // 2 + (KSH - 1) * SDPX // 2, cy - h // 2 - (KSH - 1) * SDPY)
def botA(left, h, cy): return (left + SHW // 2, cy + h // 2)

def sheet(x, yt, h, fill, stroke):
    st = (f"shape=parallelogram;perimeter=parallelogramPerimeter;html=1;fixedSize=1;"
          f"size={LEAN};fillColor={fill};strokeColor={stroke};")
    cells.append(f'<mxCell id="{nid()}" value="" style="{st}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{yt}" width="{SHW + LEAN}" height="{h}" as="geometry"/></mxCell>')

def layer(left, h, kind, cy):
    base = KIND[kind]; st = shade(base, -0.45)
    for k in range(KSH - 1, -1, -1):                 # back (k=KSH-1) first
        sheet(left + k * SDPX, cy - h // 2 - k * SDPY, h, shade(base, 0.11 * k), st)

def image_cell(x, y, w, h, data):
    cells.append(f'<mxCell id="{nid()}" value="" style="shape=image;imageAspect=0;'
                 f'image=data:image/png,{data};" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def rrect(x, y, w, h, stroke, fill="none", dashdot=False, sw=2, arc=20):
    st = f"rounded=1;arcSize={arc};html=1;strokeColor={stroke};strokeWidth={sw};"
    st += (f"fillColor={fill};" if fill != "none" else "fillColor=none;")
    if dashdot: st += DASHDOT
    cells.append(f'<mxCell id="{nid()}" value="" style="{st}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def text(x, y, w, h, s, size=12, bold=True, color=BLACK, align="left"):
    s = s.replace("\n", " ")
    st = (f"text;html=1;whiteSpace=wrap;align={align};verticalAlign=middle;"
          f"fontFamily=Helvetica;fontSize={size};fontColor={color};")
    if bold: st += "fontStyle=1;"
    cells.append(f'<mxCell id="{nid()}" value="{s}" style="{st}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def edge(x1, y1, x2, y2, points=None, stroke=BLACK, sw=1.7, dashdot=False, end="classic", extra=""):
    st = f"endArrow={end};startArrow=none;html=1;rounded=1;strokeColor={stroke};strokeWidth={sw};"
    if dashdot: st += DASHDOT
    st += extra
    pts = ("" if not points else '<Array as="points">'
           + "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in points) + "</Array>")
    cells.append(f'<mxCell id="{nid()}" style="{st}" edge="1" parent="1">'
                 f'<mxGeometry relative="1" as="geometry">'
                 f'<mxPoint x="{x1}" y="{y1}" as="sourcePoint"/>'
                 f'<mxPoint x="{x2}" y="{y2}" as="targetPoint"/>{pts}</mxGeometry></mxCell>')

def E(pts, stroke=BLACK, sw=1.7, dash=False, arrow=True, extra=""):
    edge(pts[0][0], pts[0][1], pts[-1][0], pts[-1][1], points=(pts[1:-1] or None),
         stroke=stroke, sw=sw, dashdot=dash, end=("classic" if arrow else "none"), extra=extra)
def L(pts, stroke=BLACK, sw=1.2): E(pts, stroke=stroke, sw=sw, arrow=False)

def odot(cx, cy, r=12, stroke=MBLUE):
    cells.append(f'<mxCell id="{nid()}" value="" style="ellipse;html=1;fillColor=#FFFFFF;'
                 f'strokeColor={stroke};strokeWidth=2;" vertex="1" parent="1">'
                 f'<mxGeometry x="{cx-r}" y="{cy-r}" width="{2*r}" height="{2*r}" as="geometry"/></mxCell>')
    d = 3.2
    cells.append(f'<mxCell id="{nid()}" value="" style="ellipse;html=1;fillColor={stroke};'
                 f'strokeColor={stroke};" vertex="1" parent="1">'
                 f'<mxGeometry x="{cx-d}" y="{cy-d}" width="{2*d}" height="{2*d}" as="geometry"/></mxCell>')
def ring(cx, cy, r=8, stroke=RED, sw=2.4):
    cells.append(f'<mxCell id="{nid()}" value="" style="ellipse;html=1;fillColor=none;'
                 f'strokeColor={stroke};strokeWidth={sw};" vertex="1" parent="1">'
                 f'<mxGeometry x="{cx-r}" y="{cy-r}" width="{2*r}" height="{2*r}" as="geometry"/></mxCell>')

# ===========================================================================
CY, X0, PITCH = 290, 250, 34
BOX_X, BOX_Y, BOX_W, BOX_H = 205, 156, 930, 256
rrect(BOX_X, BOX_Y, BOX_W, BOX_H, BLACK, sw=2.2, arc=12)

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
for sl in reversed(slabs):                          # left = front: emit last -> on top
    layer(sl["left"], sl["h"], sl["kind"], CY)

def by_stage(s): return [x for x in slabs if x["stage"] == s]
for es, ds, ay in [(0, 8, 178), (1, 7, 190), (2, 6, 202), (3, 5, 214)]:
    a = by_stage(es)[-1]; b = by_stage(ds)[0]
    pa = topA(a["left"], a["h"], CY); pb = topA(b["left"], b["h"], CY)
    E([pa, (pa[0], ay), (pb[0], ay), pb], stroke="#9AA0A6", sw=1.3, extra="endSize=5;")

IMG_W, IMG_H = 165, 98
IX, IY = 42, CY - IMG_H // 2
image_cell(IX, IY, IMG_W, IMG_H, IMG_INPUT)
E([(IX + IMG_W + 4, CY), (X0 + 2, CY)], sw=2.0)

MX, MY, MW, MH = 330, 50, 360, 104
rrect(MX, MY, MW, MH, MBLUE, dashdot=True, sw=2.2, arc=22)
mc = MY + MH // 2
L([(MX + 24, mc + 12), (MX + MW - 24, mc + 12)], stroke="#3A3A3A", sw=1.1)
L([(MX + MW // 2, mc + 12), (MX + MW // 2, mc - 18)], stroke="#3A3A3A", sw=1.0)
cxm = MX + MW // 2
E([(MX + 20, mc - 2), (cxm - 40, mc - 2), (cxm - 6, mc + 26), (cxm + 4, mc - 26),
   (cxm + 14, mc - 2), (MX + MW - 20, mc - 2)], stroke=RED, sw=2.2, arrow=False, extra="curved=1;")
E([(IX + 22, IY + 6), (IX + 22, mc), (MX + 6, mc)], stroke=MBLUE, sw=2.0)

# attention maps (top) as a layered-sheet group
AX, AH, AYC = 770, 72, 104
E([(MX + MW, mc), (AX - 4, AYC)], stroke=MBLUE, sw=2.0)
layer(AX, AH, "att", AYC)
ring(AX + 4, AYC, r=5)

atts = [s for s in slabs if s["kind"] == "att"]; feats = [s for s in slabs if s["kind"] == "feat"]
a_half, a_full = atts; f_half, f_full = feats
pf = topA(a_full["left"], a_full["h"], CY); ph = topA(a_half["left"], a_half["h"], CY)
E([(AX + 52, AYC), (pf[0], AYC), pf], stroke=MBLUE, sw=2.0, dash=True)
E([(ph[0], AYC), ph], stroke=MBLUE, sw=2.0, dash=True)
for f, a in [(f_half, a_half), (f_full, a_full)]:
    bf = botA(f["left"], f["h"], CY); ba = botA(a["left"], a["h"], CY)
    ox = (bf[0] + ba[0]) // 2; oy = CY + max(f["h"], a["h"]) // 2 + 24
    odot(ox, oy)
    L([bf, (bf[0], oy), (ox - 8, oy - 9)], stroke=MBLUE, sw=1.4)
    L([ba, (ba[0], oy), (ox + 8, oy - 9)], stroke=MBLUE, sw=1.4)

E([(BOX_X + BOX_W, CY), (BOX_X + BOX_W + 60, CY)], sw=2.0)
OX, OY = BOX_X + BOX_W + 62, CY - IMG_H // 2
image_cell(OX, OY, IMG_W, IMG_H, IMG_HEAT)

# ---- legend ----
LY = 452
def leg(x, s): text(x, LY - 8, 128, 46, s, size=12, bold=True)
groups = [30, 230, 430, 630, 830, 1030, 1230, 1430]
gx = groups[0]; image_cell(gx, LY, 40, 26, IMG_INPUT); leg(gx + 48, "Input frames")
gx = groups[1]; rrect(gx, LY + 4, 48, 30, BLACK, sw=2, arc=24); leg(gx + 58, "TrackNet backbone")
gx = groups[2]; rrect(gx, LY + 4, 50, 30, MBLUE, dashdot=True, sw=2, arc=26)
E([(gx + 8, LY + 19), (gx + 21, LY + 28), (gx + 27, LY + 9), (gx + 42, LY + 19)], stroke=RED, sw=1.8, arrow=False, extra="curved=1;")
leg(gx + 62, "Motion attention module")
gx = groups[3]; E([(gx + 16, LY + 2), (gx + 16, LY + 34)], stroke=MBLUE, sw=2.4, dash=True); leg(gx + 30, "Motion-aware fusion path")
gx = groups[4]; layer(gx, 30, "att", LY + 18); leg(gx + 52, "Attention maps")
gx = groups[5]; layer(gx, 30, "feat", LY + 18); leg(gx + 52, "Feature maps")
gx = groups[6]; odot(gx + 13, LY + 19, r=12); leg(gx + 34, "Element-wise multiplication")
gx = groups[7]; image_cell(gx, LY, 40, 26, IMG_HEAT); leg(gx + 48, "Predicted heatmaps")

body = "\n        ".join(cells)
xml = f'''<mxfile host="app.diagrams.net" type="device">
  <diagram name="TrackNetV4" id="tracknetv4">
    <mxGraphModel dx="1600" dy="900" grid="0" gridSize="10" guides="1" tooltips="1"
        connect="1" arrows="1" fold="1" page="1" pageScale="1"
        pageWidth="1640" pageHeight="600" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        {body}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''
out = f"{DIAG}/tracknetv4_architecture.drawio"
MD.parseString(xml)
with open(out, "w") as f:
    f.write(xml)
print("wrote", out, "| cells:", len(cells))

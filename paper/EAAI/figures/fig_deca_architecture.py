"""
fig_deca_architecture.py — DECA architecture diagram, publication-ready.

Double-column width (~6.6 in). Arrows drawn manually as two-segment
L-shapes (vertical shaft + horizontal arrowhead) for clean rendering.

Output: fig_deca_architecture.pdf  +  fig_deca_architecture.png
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Figure & font settings (Elsevier EAAI) ───────────────────────────────────
FIG_W = 3.54    # single column = 90 mm (Elsevier standard)
FS_T  = 8.0     # title bar font  (≥7 pt required)
FS_B  = 7.0     # body text       (≥7 pt required)
FS_LG = 6.0     # legend          (≥6 pt for subscripts)

plt.rcParams.update({
    "font.family":   "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":     FS_B,
    "pdf.fonttype":  42,
    "ps.fonttype":   42,
})

# ── Colours ───────────────────────────────────────────────────────────────────
C_IO   = "#F2F3F4"
C_ENC  = "#D6EAF8"
C_ENC2 = "#D5F5E3"
C_CTX  = "#EAF2FF"
C_CTX2 = "#EAFAF1"
C_CA   = "#FDEBD0"
C_GF   = "#D5F5E3"
C_CLS  = "#FADBD8"
C_EDGE = "#2C3E50"
C_ARR  = "#2C3E50"

def _tc(c):
    """Slightly darker title-bar colour."""
    h = c.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    f = 0.82
    return f"#{int(r*f):02x}{int(g*f):02x}{int(b*f):02x}"

# ── Layout (inches, y=0 at bottom) ───────────────────────────────────────────
MX    = 0.04    # left/right margin
HGAP  = 0.28    # gap between the two branch columns
BW    = (FIG_W - 2*MX - HGAP) / 2   # branch block width  ≈ 2.725 in
BW_M  = BW*2 + HGAP                  # merged block width  ≈ 6.00 in

left_x   = MX
right_x  = MX + BW + HGAP
merged_x = MX

# Block heights
H_IO  = 0.40
H_ENC = 0.65
H_CTX = 0.40
H_CA  = 0.55
H_GF  = 0.40
H_CLS = 0.40
H_LEG = 0.44   # 2-row legend height

VGAP = 0.12

# Y positions (bottom to top, building from bottom)
y_leg  = 0.05
y_cls  = y_leg + H_LEG + 0.08
y_gf   = y_cls  + H_CLS + VGAP
y_ca   = y_gf   + H_GF  + VGAP
y_ctx  = y_ca   + H_CA  + VGAP
y_enc  = y_ctx  + H_CTX + VGAP
y_inp  = y_enc  + H_ENC + VGAP

# Figure height fits content exactly (small top margin)
FIG_H = y_inp + H_IO + 0.08


# ── Helpers ───────────────────────────────────────────────────────────────────
def draw_block(ax, x, y, w, h, title, lines, fill):
    """Draw titled block. Returns (top_y, bottom_y, center_x)."""
    title_h = h * 0.30
    body_h  = h - title_h
    kw = dict(linewidth=0.7, edgecolor=C_EDGE, zorder=3)
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.005", facecolor=fill, **kw))
    ax.add_patch(FancyBboxPatch((x, y + body_h), w, title_h,
                 boxstyle="round,pad=0.005", facecolor=_tc(fill), **kw))
    cx = x + w / 2
    ax.text(cx, y + body_h + title_h/2, title,
            ha="center", va="center", fontsize=FS_T,
            fontweight="bold", color="#111111", zorder=4)
    n = len(lines)
    for i, line in enumerate(lines):
        ly = y + body_h * (n - i - 0.5) / n
        ax.text(cx, ly, line,
                ha="center", va="center", fontsize=FS_B,
                color="#222222", zorder=4)
    return y + h, y, cx


def arrow_v(ax, x, y_start, y_end, color=C_ARR):
    """Straight downward arrow from y_start to y_end  (y_start > y_end)."""
    ax.annotate("", xy=(x, y_end), xytext=(x, y_start),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=0.9, mutation_scale=9),
                zorder=6)




# ── Create figure ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect("auto")
ax.axis("off")

# ── INPUT BOXES ──────────────────────────────────────────────────────────────
_, _, cx_l = draw_block(ax, left_x,  y_inp, BW, H_IO,
    "Technical Features", ["x_T   (15 features)"], C_IO)
_, _, cx_r = draw_block(ax, right_x, y_inp, BW, H_IO,
    "On-chain Features",  ["x_O   (18 features)"], C_IO)

# ── MLP ENCODERS ─────────────────────────────────────────────────────────────
enc_lines = ["4 ResNet layers,  d=128", "GELU  |  LayerNorm  |  Dropout"]
draw_block(ax, left_x,  y_enc, BW, H_ENC, "MLP Encoder  T", enc_lines, C_ENC)
draw_block(ax, right_x, y_enc, BW, H_ENC, "MLP Encoder  O", enc_lines, C_ENC2)

# ── CONTEXT TOKEN BOXES ───────────────────────────────────────────────────────
draw_block(ax, left_x,  y_ctx, BW, H_CTX,
           "+ K=4 Context Tokens", ["S_T   (5 x 128)"], C_CTX)
draw_block(ax, right_x, y_ctx, BW, H_CTX,
           "+ K=4 Context Tokens", ["S_O   (5 x 128)"], C_CTX2)

# ── CROSS-ATTENTION ───────────────────────────────────────────────────────────
_, _, cx_m = draw_block(ax, merged_x, y_ca, BW_M, H_CA,
    "Bidirectional Cross-Attention",
    ["Q=S_T, K=V=S_O   <-->   Q=S_O, K=V=S_T",
     "H = 4 heads   |   N_ca = 1 layer"], C_CA)

# ── GATED FUSION ─────────────────────────────────────────────────────────────
draw_block(ax, merged_x, y_gf, BW_M, H_GF,
    "Gated Fusion",
    ["g = sigmoid(Wg[z_T ; z_O]),   f = g * Wf[z_T ; z_O]"], C_GF)

# ── CLASSIFICATION HEAD ───────────────────────────────────────────────────────
draw_block(ax, merged_x, y_cls, BW_M, H_CLS,
    "FC + Sigmoid",
    ["y_hat = P(y = 1 | x)"], C_CLS)

# ── ARROWS (all uniform straight downward arrows) ────────────────────────────
# ASINK: arrowhead sinks this far inside the destination block
# ASHIFT: both endpoints shifted upward by 2 px @ 200 dpi = 0.01 in
ASINK  = 0.04
ASHIFT = 0.015

# Input (bottom) → Encoder (top)
arrow_v(ax, cx_l, y_inp   + ASHIFT, y_enc + H_ENC - ASINK + ASHIFT)
arrow_v(ax, cx_r, y_inp   + ASHIFT, y_enc + H_ENC - ASINK + ASHIFT)
# Encoder (bottom) → Context tokens (top)
arrow_v(ax, cx_l, y_enc   + ASHIFT, y_ctx + H_CTX - ASINK + ASHIFT)
arrow_v(ax, cx_r, y_enc   + ASHIFT, y_ctx + H_CTX - ASINK + ASHIFT)
# Context tokens (bottom) → Cross-Attention (top)
arrow_v(ax, cx_l, y_ctx   + ASHIFT, y_ca  + H_CA  - ASINK + ASHIFT)
arrow_v(ax, cx_r, y_ctx   + ASHIFT, y_ca  + H_CA  - ASINK + ASHIFT)

# S_T / S_O labels alongside the arrows above CA
mid_y_ctx_ca = (y_ctx + y_ca + H_CA) / 2
ax.text(cx_l - 0.08, mid_y_ctx_ca,
        "S_T", ha="right", va="center", fontsize=7.0, color="#7F8C8D", zorder=6)
ax.text(cx_r + 0.08, mid_y_ctx_ca,
        "S_O", ha="left",  va="center", fontsize=7.0, color="#7F8C8D", zorder=6)

# CA (bottom) → Gated Fusion (top) → Classifier (top)
arrow_v(ax, cx_m, y_ca  + ASHIFT, y_gf  + H_GF  - ASINK + ASHIFT)
arrow_v(ax, cx_m, y_gf  + ASHIFT, y_cls + H_CLS - ASINK + ASHIFT)

# ── LEGEND  (2 rows × 3 cols, centred) ───────────────────────────────────────
ax.axhline(y_leg + H_LEG + 0.01, color="#CCCCCC", lw=0.6, zorder=2)

legend_items = [
    (C_IO,   "Input"),
    (C_ENC,  "Tech encoder"),
    (C_ENC2, "On-chain encoder"),
    (C_CA,   "Cross-attention"),
    (C_GF,   "Gated fusion"),
    (C_CLS,  "Classifier"),
]

NCOLS   = 3
sw      = 0.10                          # square side
gapS    = 0.04                          # square → text
row_h   = H_LEG / 2                     # height per row
slot_w  = (FIG_W - 2*MX) / NCOLS       # equal column slots, guaranteed centred

for row in range(2):
    for col in range(NCOLS):
        idx     = row * NCOLS + col
        c, lbl  = legend_items[idx]
        x_cur   = MX + col * slot_w
        sy      = y_leg + (1 - row) * row_h + (row_h - sw) / 2
        ax.add_patch(FancyBboxPatch((x_cur, sy), sw, sw,
                     boxstyle="round,pad=0.003",
                     facecolor=c, edgecolor="#888888", linewidth=0.5, zorder=3))
        ax.text(x_cur + sw + gapS, sy + sw / 2, lbl,
                ha="left", va="center", fontsize=FS_LG, color="#333333", zorder=4)

# ── SAVE ─────────────────────────────────────────────────────────────────────
out_dir = Path(__file__).parent
fig.savefig(out_dir / "fig_deca_architecture.pdf", bbox_inches="tight", pad_inches=0, dpi=300)
fig.savefig(out_dir / "fig_deca_architecture.png", bbox_inches="tight", pad_inches=0, dpi=1000)
plt.close(fig)
print(f"[saved] {out_dir / 'fig_deca_architecture.pdf'}")
print(f"[saved] {out_dir / 'fig_deca_architecture.png'}")

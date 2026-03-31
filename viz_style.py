from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource

PALETTE = {
    "fig_bg": "#0b0c10",
    "panel_bg": "#11141d",
    "panel_edge": "#1f2a38",
    "text": "#e2e8f0",
    "subtext": "#9fb3c8",
    "trace_base": "#5de0ff",
    "trace_pert": "#ffb36b",
    "tool_main": "#ff4d4d",
    "tool_glow": "#ff9a9a",
    "stock_2d": "#22303d",
    "removed_2d": "#1c2b36",
    "height_top": "#1f3a4d",
    "height_floor": "#101822",
    "contour": "#7cc7ff",
}


def style_axis(ax, title: str, width: float, height: float) -> None:
    ax.set_xlim(-0.3, width + 0.3)
    ax.set_ylim(-0.3, height + 0.3)
    ax.set_aspect("equal")
    ax.set_facecolor(PALETTE["panel_bg"])
    ax.grid(True, alpha=0.28, color="#2a3645", linestyle=":", linewidth=0.7)
    ax.set_title(title, color=PALETTE["text"], fontsize=11.5, fontweight="bold", pad=6)
    ax.tick_params(colors=PALETTE["subtext"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["panel_edge"])
        spine.set_linewidth(1.0)


def draw_tool_marker(ax, x: float, y: float, r: float) -> None:
    core = plt.Circle((x, y), r * 0.94, facecolor=PALETTE["tool_main"], edgecolor="white", linewidth=1.2, alpha=0.95, zorder=6)
    rim = plt.Circle((x, y), r, facecolor=(0, 0, 0, 0), edgecolor="white", linewidth=1.4, alpha=0.9, zorder=7)
    glow = plt.Circle((x, y), r * 1.35, facecolor=PALETTE["tool_glow"], edgecolor="none", alpha=0.18, zorder=5)
    ax.add_patch(glow)
    ax.add_patch(core)
    ax.add_patch(rim)


def shaded_height_rgb(z_top: np.ndarray, pocket_depth: float) -> np.ndarray:
    removed_mask = z_top <= -pocket_depth + 1e-9
    base = np.zeros(z_top.shape + (3,), dtype=float)
    base[..., 0] = np.where(removed_mask, 0.25, 0.16)
    base[..., 1] = np.where(removed_mask, 0.48, 0.28)
    base[..., 2] = np.where(removed_mask, 0.62, 0.35)
    ls = LightSource(azdeg=320, altdeg=42)
    return ls.shade_rgb(base, z_top, fraction=0.92)


def style_axis3d(ax, title: str, width: float, height: float, pocket_depth: float) -> None:
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_zlim(-pocket_depth - 0.5, 0.8)
    ax.view_init(elev=28, azim=-55)
    ax.set_title(title, color=PALETTE["text"], fontsize=11, pad=6)
    ax.tick_params(colors=PALETTE["subtext"], labelsize=7)
    ax.set_box_aspect([1, 1, 0.45])
    ax.xaxis.pane.set_facecolor(PALETTE["panel_bg"])
    ax.yaxis.pane.set_facecolor(PALETTE["panel_bg"])
    ax.zaxis.pane.set_facecolor(PALETTE["panel_bg"])
    ax.w_xaxis.line.set_color(PALETTE["panel_edge"])
    ax.w_yaxis.line.set_color(PALETTE["panel_edge"])
    ax.w_zaxis.line.set_color(PALETTE["panel_edge"])
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.label.set_color(PALETTE["text"])
        for tick in axis.get_ticklines():
            tick.set_color(PALETTE["subtext"])


def add_tool_inset(ax, tool_spec: Mapping[str, float | int | str], tool_png: Path | None) -> None:
    inset = ax.inset_axes([0.62, 0.54, 0.35, 0.42], zorder=20)
    inset.set_xlim(0.0, 1.0)
    inset.set_ylim(0.0, 1.0)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor((0.08, 0.1, 0.14, 0.92))
    for spine in inset.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color(PALETTE["panel_edge"])
    if tool_png and tool_png.exists():
        img = plt.imread(tool_png)
        inset.imshow(img, extent=[0.04, 0.48, 0.08, 0.95], aspect="auto")
    inset.text(0.52, 0.95, "Tool Inset", fontsize=7.6, fontweight="bold", ha="left", va="top", color=PALETTE["text"])
    inset.text(0.52, 0.79, f"D: {tool_spec['diameter_mm']:.1f} mm", fontsize=6.8, ha="left", color=PALETTE["text"])
    inset.text(0.52, 0.64, f"Z: {int(tool_spec['flutes'])}", fontsize=6.8, ha="left", color=PALETTE["text"])
    inset.text(0.52, 0.49, f"Helix: {tool_spec['helix_deg']:.0f} deg", fontsize=6.8, ha="left", color=PALETTE["text"])

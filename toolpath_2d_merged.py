from pathlib import Path
import subprocess
import sys
import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle

from toolpath_core import (
    PocketSpec,
    build_disk_offsets,
    carve_2d,
    center_to_index,
    create_material_grid,
    pocket_paths,
)


OUTPUT_GIF = Path(__file__).with_name("toolpath_2d_merged.gif")
TOOL_RENDER_PNG = Path(__file__).with_name("tool_model_render.png")
TOOL_RENDER_STL = Path(__file__).with_name("tool_model_render.stl")
TOOL_RENDER_SCRIPT = Path(__file__).with_name("tool_model_cadquery_render.py")
TOOL_RENDER_META = Path(__file__).with_name("tool_model_render.meta.json")

MIN_TOOL_DIAMETER_MM = 10.0
PATH_SAMPLES = 220
FRAME_COUNT = 56
FPS = 14
CELL_MM_2D = 0.08

PALETTE = {
    "fig_bg": "#ffffff",
    "panel_bg": "#ffffff",
    "panel_edge": "#d1d5db",
    "text": "#111827",
    "subtext": "#4b5563",
    "trace_base": "#000000",
    "trace_pert": "#1f77b4",
    "tool_main": "#c53030",
    "stock_2d": "#f4d79a",
    "removed_2d": "#a9a9a9",
    "contour": "#5a5a5a",
}

TOOL_CONFIG = {
    "type": "Flat End Mill",
    "flutes": 3,
    "flute_length_mm": 25.0,
    "stickout_mm": 72.0,
    "helix_deg": 40.0,
    "tool_material": "Carbide",
}


@dataclass(frozen=True)
class RenderSpec:
    width: float = 50.0
    height: float = 50.0
    stepover: float = 5.0
    samples: int = PATH_SAMPLES
    perturb_amp: float = 0.18
    perturb_wavelength: float = 1.4


def resolve_tool_spec(stepover: float) -> dict[str, float | int | str]:
    spec = dict(TOOL_CONFIG)
    spec["diameter_mm"] = float(max(MIN_TOOL_DIAMETER_MM, 2.0 * stepover))
    return spec


def ensure_tool_render(tool_spec: dict[str, float | int | str]) -> None:
    if not TOOL_RENDER_SCRIPT.exists():
        return

    desired_meta = {
        "tool_type": str(tool_spec["type"]),
        "diameter_mm": float(tool_spec["diameter_mm"]),
        "flutes": int(tool_spec["flutes"]),
        "flute_length_mm": float(tool_spec["flute_length_mm"]),
        "stickout_mm": float(tool_spec["stickout_mm"]),
        "helix_deg": float(tool_spec["helix_deg"]),
        "tool_material": str(tool_spec["tool_material"]),
    }

    existing_meta = None
    if TOOL_RENDER_META.exists():
        try:
            existing_meta = json.loads(TOOL_RENDER_META.read_text(encoding="utf-8"))
        except Exception:
            existing_meta = None

    needs_regen = (
        (not TOOL_RENDER_PNG.exists())
        or (TOOL_RENDER_SCRIPT.stat().st_mtime > TOOL_RENDER_PNG.stat().st_mtime)
        or (existing_meta != desired_meta)
    )
    if not needs_regen:
        return

    cmd = [
        sys.executable,
        str(TOOL_RENDER_SCRIPT),
        "--tool-type",
        str(tool_spec["type"]),
        "--diameter-mm",
        f"{float(tool_spec['diameter_mm']):.6f}",
        "--flutes",
        str(int(tool_spec["flutes"])),
        "--flute-length-mm",
        f"{float(tool_spec['flute_length_mm']):.6f}",
        "--stickout-mm",
        f"{float(tool_spec['stickout_mm']):.6f}",
        "--helix-deg",
        f"{float(tool_spec['helix_deg']):.6f}",
        "--tool-material",
        str(tool_spec["tool_material"]),
        "--png",
        str(TOOL_RENDER_PNG),
        "--stl",
        str(TOOL_RENDER_STL),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception as exc:
        print(f"Tool render generation failed: {exc}")
        return

    if TOOL_RENDER_PNG.exists():
        try:
            TOOL_RENDER_META.write_text(json.dumps(desired_meta, indent=2), encoding="utf-8")
        except Exception:
            pass


def style_axis(ax, title: str, width: float, height: float) -> None:
    ax.set_xlim(-0.4, width + 0.4)
    ax.set_ylim(-0.4, height + 0.4)
    ax.set_aspect("equal")
    ax.set_facecolor(PALETTE["panel_bg"])
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title, color=PALETTE["text"], fontsize=11.5, fontweight="bold", pad=6)
    ax.tick_params(colors=PALETTE["subtext"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["panel_edge"])
        spine.set_linewidth(1.0)


def add_tool_inset(ax, tool_spec: dict[str, float | int | str]) -> None:
    inset = ax.inset_axes([0.60, 0.56, 0.37, 0.41], zorder=30)
    inset.set_xlim(0.0, 1.0)
    inset.set_ylim(0.0, 1.0)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor((0.97, 0.98, 1.0, 0.98))
    for spine in inset.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color(PALETTE["panel_edge"])
    if TOOL_RENDER_PNG.exists():
        img = plt.imread(TOOL_RENDER_PNG)
        inset.imshow(img, extent=[0.04, 0.48, 0.08, 0.95], aspect="auto")
    inset.text(0.52, 0.95, "Tool Used", fontsize=7.6, fontweight="bold", ha="left", va="top", color=PALETTE["text"])
    inset.text(0.52, 0.79, f"D: {tool_spec['diameter_mm']:.1f} mm", fontsize=6.8, ha="left", color=PALETTE["text"])
    inset.text(0.52, 0.64, f"Z: {int(tool_spec['flutes'])}", fontsize=6.8, ha="left", color=PALETTE["text"])
    inset.text(0.52, 0.49, f"Helix: {tool_spec['helix_deg']:.0f} deg", fontsize=6.8, ha="left", color=PALETTE["text"])


def build_paths(spec: RenderSpec) -> tuple[np.ndarray, np.ndarray]:
    core_spec = PocketSpec(
        width=spec.width,
        height=spec.height,
        stepover=spec.stepover,
        samples=spec.samples,
        perturb_amplitude=spec.perturb_amp,
        perturb_wavelength=spec.perturb_wavelength,
    )
    base_path, perturbed_path, _ = pocket_paths(core_spec)
    return base_path, perturbed_path


def build_gif() -> None:
    spec = RenderSpec()
    tool_spec = resolve_tool_spec(spec.stepover)
    ensure_tool_render(tool_spec)
    tool_radius_mm = float(tool_spec["diameter_mm"]) / 2.0

    base_path, perturbed_path = build_paths(spec)
    frame_indices = np.linspace(0, len(base_path) - 1, FRAME_COUNT, dtype=int)
    frame_indices = np.unique(frame_indices)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    fig.patch.set_facecolor(PALETTE["fig_bg"])

    xx_base, yy_base, remaining_base = create_material_grid(spec.width, spec.height, cell_mm=CELL_MM_2D)
    xx_pert, yy_pert, remaining_pert = create_material_grid(spec.width, spec.height, cell_mm=CELL_MM_2D)
    cell_x = float(xx_base[0, 1] - xx_base[0, 0])
    cell_y = float(yy_base[1, 0] - yy_base[0, 0])
    x0 = float(xx_base[0, 0])
    y0 = float(yy_base[0, 0])
    ox, oy = build_disk_offsets(min(cell_x, cell_y), tool_radius_mm)
    cmap = ListedColormap([PALETTE["removed_2d"], PALETTE["stock_2d"]])

    for ax, title in [
        (axes[0], "2D Plan View - Standard"),
        (axes[1], "2D Plan View - Adapted"),
    ]:
        style_axis(ax, title, spec.width, spec.height)

    base_mat = axes[0].imshow(remaining_base, origin="lower", extent=[0.0, spec.width, 0.0, spec.height], cmap=cmap, vmin=0, vmax=1, alpha=0.9)
    pert_mat = axes[1].imshow(remaining_pert, origin="lower", extent=[0.0, spec.width, 0.0, spec.height], cmap=cmap, vmin=0, vmax=1, alpha=0.9)
    base_line, = axes[0].plot([], [], color=PALETTE["trace_base"], linewidth=2.0)
    pert_line, = axes[1].plot([], [], color=PALETTE["trace_pert"], linewidth=2.0)
    base_circle = Circle((0.0, 0.0), tool_radius_mm, fill=False, color=PALETTE["tool_main"], linewidth=1.6)
    pert_circle = Circle((0.0, 0.0), tool_radius_mm, fill=False, color=PALETTE["tool_main"], linewidth=1.6)
    axes[0].add_patch(base_circle)
    axes[1].add_patch(pert_circle)
    base_contour = [None]
    pert_contour = [None]
    contour_stride = 2
    add_tool_inset(axes[1], tool_spec)
    fig.suptitle("Toolpath Comparison: Standard (Left) vs Adapted (Right)", fontsize=13, fontweight="bold")

    def init():
        base_mat.set_data(remaining_base)
        pert_mat.set_data(remaining_pert)
        base_line.set_data([], [])
        pert_line.set_data([], [])
        base_circle.center = (0.0, 0.0)
        pert_circle.center = (0.0, 0.0)
        return base_mat, pert_mat, base_line, pert_line, base_circle, pert_circle

    def update(frame_pos: int):
        frame_idx = int(frame_indices[frame_pos])
        bx = float(base_path[frame_idx, 0])
        by = float(base_path[frame_idx, 1])
        px = float(perturbed_path[frame_idx, 0])
        py = float(perturbed_path[frame_idx, 1])

        base_ix, base_iy = center_to_index(bx, by, x0, y0, cell_x, cell_y)
        pert_ix, pert_iy = center_to_index(px, py, x0, y0, cell_x, cell_y)
        carve_2d(remaining_base, base_ix, base_iy, ox, oy)
        carve_2d(remaining_pert, pert_ix, pert_iy, ox, oy)
        base_mat.set_data(remaining_base)
        pert_mat.set_data(remaining_pert)

        if frame_pos % contour_stride == 0 or frame_pos == len(frame_indices) - 1:
            if base_contour[0] is not None:
                base_contour[0].remove()
            if pert_contour[0] is not None:
                pert_contour[0].remove()
            base_contour[0] = axes[0].contour(xx_base, yy_base, (remaining_base == 0).astype(float), levels=[0.5], colors=PALETTE["contour"], linewidths=0.9)
            pert_contour[0] = axes[1].contour(xx_pert, yy_pert, (remaining_pert == 0).astype(float), levels=[0.5], colors=PALETTE["contour"], linewidths=0.9)

        base_line.set_data(base_path[: frame_idx + 1, 0], base_path[: frame_idx + 1, 1])
        pert_line.set_data(perturbed_path[: frame_idx + 1, 0], perturbed_path[: frame_idx + 1, 1])
        base_circle.center = (bx, by)
        pert_circle.center = (px, py)
        return base_mat, pert_mat, base_line, pert_line, base_circle, pert_circle

    anim = FuncAnimation(fig, update, init_func=init, frames=len(frame_indices), interval=1000 / FPS, blit=False)
    anim.save(OUTPUT_GIF, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"Created {OUTPUT_GIF}")

def main() -> None:
    build_gif()


if __name__ == "__main__":
    main()


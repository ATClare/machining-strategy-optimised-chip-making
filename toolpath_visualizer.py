from pathlib import Path
import subprocess
import sys
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle

from toolpath_core import (
    PocketSpec,
    apply_sinusoidal_perturbation,
    build_disk_offsets,
    build_pocket_raster,
    carve_2d,
    center_to_index,
    create_material_grid,
    resample_polyline,
)


OUTPUT_GIF = Path(__file__).with_name("pocket_toolpath_comparison.gif")
MIN_TOOL_DIAMETER_MM = 10.0
TOOL_RENDER_PNG = Path(__file__).with_name("tool_model_render.png")
TOOL_RENDER_STL = Path(__file__).with_name("tool_model_render.stl")
TOOL_RENDER_SCRIPT = Path(__file__).with_name("tool_model_cadquery_render.py")
TOOL_RENDER_META = Path(__file__).with_name("tool_model_render.meta.json")


TOOL_CONFIG = {
    "type": "Flat End Mill",
    "flutes": 3,
    "flute_length_mm": 25.0,
    "stickout_mm": 72.0,
    "helix_deg": 40.0,
    "tool_material": "Carbide",
}


def resolve_tool_spec(diameter_mm: float) -> dict[str, float | int | str]:
    spec = dict(TOOL_CONFIG)
    spec["diameter_mm"] = float(diameter_mm)
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


def add_tool_inset(ax, tool_spec: dict[str, float | int | str]) -> None:
    inset = ax.inset_axes([0.60, 0.56, 0.37, 0.41], zorder=30)
    inset.set_xlim(0.0, 1.0)
    inset.set_ylim(0.0, 1.0)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor((0.97, 0.98, 1.0, 0.98))
    for spine in inset.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#9ca3af")
    if TOOL_RENDER_PNG.exists():
        img = plt.imread(TOOL_RENDER_PNG)
        inset.imshow(img, extent=[0.04, 0.48, 0.08, 0.95], aspect="auto")
    inset.text(0.52, 0.95, "Tool Used", fontsize=7.6, fontweight="bold", ha="left", va="top", color="#111827")
    inset.text(0.52, 0.79, f"D: {tool_spec['diameter_mm']:.1f} mm", fontsize=6.8, ha="left", color="#111827")
    inset.text(0.52, 0.64, f"Z: {int(tool_spec['flutes'])}", fontsize=6.8, ha="left", color="#111827")
    inset.text(0.52, 0.49, f"Helix: {tool_spec['helix_deg']:.0f} deg", fontsize=6.8, ha="left", color="#111827")


def estimate_chip_metrics_7075(
    tool_diameter_mm: float,
    flutes: int,
    spindle_rpm: float,
    feed_mm_min: float,
    axial_doc_mm: float,
    radial_doc_mm: float,
    perturb_wavelength_mm: float,
) -> dict[str, float]:
    if flutes <= 0 or spindle_rpm <= 0:
        raise ValueError("flutes and spindle_rpm must be > 0")

    ae = min(max(radial_doc_mm, 1e-9), tool_diameter_mm)
    entry_angle = float(np.arccos(1.0 - 2.0 * ae / tool_diameter_mm))
    fz = feed_mm_min / (spindle_rpm * flutes)
    h_max = fz * np.sin(entry_angle)
    h_mean = fz * (1.0 - np.cos(entry_angle)) / max(entry_angle, 1e-9)
    mrr = feed_mm_min * axial_doc_mm * radial_doc_mm

    # Geometric chip contact/chip length estimate along engaged cutting arc.
    chip_contact_length = 0.5 * tool_diameter_mm * entry_angle

    # Heuristic "break interval": periodic waviness can encourage shorter chip segmentation.
    chip_break_length_est = max(perturb_wavelength_mm / 2.0, 1e-6)
    chip_segment_length_est = min(chip_contact_length, chip_break_length_est)

    return {
        "fz_mm_per_tooth": fz,
        "h_max_mm": h_max,
        "h_mean_mm": h_mean,
        "mrr_mm3_per_min": mrr,
        "chip_contact_length_mm": chip_contact_length,
        "chip_break_length_est_mm": chip_break_length_est,
        "chip_segment_length_est_mm": chip_segment_length_est,
    }


def animate_paths(
    base_path: np.ndarray,
    perturbed_path: np.ndarray,
    metrics: dict[str, float],
    tool_diameter_mm: float,
    fps: int = 24,
) -> None:
    frame_count = len(base_path)
    width = max(base_path[:, 0].max(), perturbed_path[:, 0].max())
    height = max(base_path[:, 1].max(), perturbed_path[:, 1].max())
    tool_radius_mm = tool_diameter_mm / 2.0
    tool_spec = resolve_tool_spec(tool_diameter_mm)
    ensure_tool_render(tool_spec)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    xx_base, yy_base, remaining_base = create_material_grid(width, height, cell_mm=0.06)
    xx_pert, yy_pert, remaining_pert = create_material_grid(width, height, cell_mm=0.06)
    cell_x = float(xx_base[0, 1] - xx_base[0, 0])
    cell_y = float(yy_base[1, 0] - yy_base[0, 0])
    x0 = float(xx_base[0, 0])
    y0 = float(yy_base[0, 0])
    ox, oy = build_disk_offsets(min(cell_x, cell_y), tool_radius_mm)
    # 0 = removed (gray), 1 = remaining stock (warm metal tone)
    cmap = ListedColormap(["#a9a9a9", "#f4d79a"])

    for ax in axes:
        ax.set_xlim(-0.4, width + 0.4)
        ax.set_ylim(-0.4, height + 0.4)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].set_title("Base Pocket Toolpath")
    axes[1].set_title("Sinusoidal Perturbed Toolpath")

    base_mat = axes[0].imshow(
        remaining_base,
        origin="lower",
        extent=[0.0, width, 0.0, height],
        cmap=cmap,
        vmin=0,
        vmax=1,
        alpha=0.85,
    )
    pert_mat = axes[1].imshow(
        remaining_pert,
        origin="lower",
        extent=[0.0, width, 0.0, height],
        cmap=cmap,
        vmin=0,
        vmax=1,
        alpha=0.85,
    )

    base_line, = axes[0].plot([], [], color="black", linewidth=2.0)
    pert_line, = axes[1].plot([], [], color="#1f77b4", linewidth=2.0)
    base_circle = Circle((0.0, 0.0), tool_radius_mm, fill=False, color="#d62728", linewidth=1.6)
    pert_circle = Circle((0.0, 0.0), tool_radius_mm, fill=False, color="#d62728", linewidth=1.6)
    axes[0].add_patch(base_circle)
    axes[1].add_patch(pert_circle)
    base_contour = [None]
    pert_contour = [None]
    contour_stride = 3

    axes[0].text(
        0.02,
        0.98,
        "Operation: Pocket raster\nTan = remaining stock, gray = removed",
        transform=axes[0].transAxes,
        va="top",
        fontsize=10,
    )
    axes[1].text(
        0.02,
        0.98,
        "Operation: Pocket + sinusoid\nTan = remaining stock, gray = removed",
        transform=axes[1].transAxes,
        va="top",
        fontsize=10,
    )
    metrics_text = (
        f"Al7075 estimate\n"
        f"fz={metrics['fz_mm_per_tooth']:.3f} mm/tooth\n"
        f"h_max={metrics['h_max_mm']:.3f} mm\n"
        f"h_mean={metrics['h_mean_mm']:.3f} mm\n"
        f"L_contact={metrics['chip_contact_length_mm']:.2f} mm\n"
        f"L_segment~{metrics['chip_segment_length_est_mm']:.2f} mm\n"
        f"MRR={metrics['mrr_mm3_per_min']:.0f} mm^3/min\n"
        f"break~{metrics['chip_break_length_est_mm']:.2f} mm"
    )
    axes[1].text(
        0.98,
        0.02,
        metrics_text,
        transform=axes[1].transAxes,
        va="bottom",
        ha="right",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#9ca3af", "alpha": 0.88},
    )
    add_tool_inset(axes[1], tool_spec)

    def init():
        base_mat.set_data(remaining_base)
        pert_mat.set_data(remaining_pert)
        base_line.set_data([], [])
        pert_line.set_data([], [])
        base_circle.center = (0.0, 0.0)
        pert_circle.center = (0.0, 0.0)
        return base_mat, pert_mat, base_line, pert_line, base_circle, pert_circle

    def update(frame: int):
        bx = float(base_path[frame, 0])
        by = float(base_path[frame, 1])
        px = float(perturbed_path[frame, 0])
        py = float(perturbed_path[frame, 1])

        base_ix, base_iy = center_to_index(bx, by, x0, y0, cell_x, cell_y)
        pert_ix, pert_iy = center_to_index(px, py, x0, y0, cell_x, cell_y)
        carve_2d(remaining_base, base_ix, base_iy, ox, oy)
        carve_2d(remaining_pert, pert_ix, pert_iy, ox, oy)
        base_mat.set_data(remaining_base)
        pert_mat.set_data(remaining_pert)
        removed_base = (remaining_base == 0).astype(float)
        removed_pert = (remaining_pert == 0).astype(float)

        if frame % contour_stride == 0 or frame == frame_count - 1:
            if base_contour[0] is not None:
                base_contour[0].remove()
            if pert_contour[0] is not None:
                pert_contour[0].remove()
            base_contour[0] = axes[0].contour(xx_base, yy_base, removed_base, levels=[0.5], colors="#5a5a5a", linewidths=0.9)
            pert_contour[0] = axes[1].contour(xx_pert, yy_pert, removed_pert, levels=[0.5], colors="#5a5a5a", linewidths=0.9)

        base_line.set_data(base_path[: frame + 1, 0], base_path[: frame + 1, 1])
        pert_line.set_data(perturbed_path[: frame + 1, 0], perturbed_path[: frame + 1, 1])
        base_circle.center = (bx, by)
        pert_circle.center = (px, py)
        return base_mat, pert_mat, base_line, pert_line, base_circle, pert_circle

    anim = FuncAnimation(fig, update, init_func=init, frames=frame_count, interval=1000 / fps, blit=False)
    anim.save(OUTPUT_GIF, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Created {OUTPUT_GIF}")


def main() -> None:
    # Pocket geometry and path settings (units can be mm).
    spec = PocketSpec()
    width = spec.width
    height = spec.height
    stepover = spec.stepover
    samples = spec.samples
    amplitude = spec.perturb_amplitude
    wavelength = spec.perturb_wavelength

    # Example cutting conditions (update these to your real process window).
    # Cutter rule: diameter is 2x stepover with a hard 10 mm minimum.
    tool_diameter_mm = max(MIN_TOOL_DIAMETER_MM, 2.0 * stepover)
    flutes = 3
    spindle_rpm = 10000.0
    feed_mm_min = 1800.0
    axial_doc_mm = 3.0
    radial_doc_mm = stepover

    coarse_base = build_pocket_raster(width=width, height=height, stepover=stepover)
    base_path, s_values = resample_polyline(coarse_base, samples=samples)
    perturbed_path = apply_sinusoidal_perturbation(
        base_path,
        s_values=s_values,
        amplitude=amplitude,
        wavelength=wavelength,
    )

    metrics = estimate_chip_metrics_7075(
        tool_diameter_mm=tool_diameter_mm,
        flutes=flutes,
        spindle_rpm=spindle_rpm,
        feed_mm_min=feed_mm_min,
        axial_doc_mm=axial_doc_mm,
        radial_doc_mm=radial_doc_mm,
        perturb_wavelength_mm=wavelength,
    )
    print("Estimated chip metrics for Al 7075 (first-pass model):")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    animate_paths(base_path, perturbed_path, metrics=metrics, tool_diameter_mm=tool_diameter_mm)


if __name__ == "__main__":
    main()

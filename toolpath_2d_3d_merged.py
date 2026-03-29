from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle


OUTPUT_GIF = Path(__file__).with_name("toolpath_2d_3d_merged.gif")


TOOL_CONFIG = {
    "type": "Flat End Mill",
    "flutes": 3,
    "flute_length_mm": 12.0,
    "stickout_mm": 22.0,
    "helix_deg": 35.0,
    "tool_material": "Carbide",
}


def build_pocket_raster(width: float, height: float, stepover: float) -> np.ndarray:
    y_levels = np.arange(0.0, height + 1e-9, stepover)
    if y_levels[-1] < height:
        y_levels = np.append(y_levels, height)

    points: list[tuple[float, float]] = []
    for i, y in enumerate(y_levels):
        if i % 2 == 0:
            points.append((0.0, y))
            points.append((width, y))
        else:
            points.append((width, y))
            points.append((0.0, y))
    return np.asarray(points, dtype=float)


def resample_polyline(path: np.ndarray, samples: int) -> np.ndarray:
    deltas = np.diff(path, axis=0)
    seg_lengths = np.sqrt((deltas**2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumulative[-1]
    s_targets = np.linspace(0.0, total, samples)
    x = np.interp(s_targets, cumulative, path[:, 0])
    y = np.interp(s_targets, cumulative, path[:, 1])
    return np.column_stack([x, y])


def apply_sinusoidal_perturbation(path: np.ndarray, amplitude: float, wavelength: float) -> np.ndarray:
    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])
    norm = np.sqrt(dx**2 + dy**2) + 1e-12
    tx = dx / norm
    ty = dy / norm
    nx = -ty
    ny = tx
    seg = np.sqrt(np.diff(path[:, 0]) ** 2 + np.diff(path[:, 1]) ** 2)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    signal = amplitude * np.sin(2.0 * np.pi * s / wavelength)
    x = path[:, 0] + signal * nx
    y = path[:, 1] + signal * ny
    return np.column_stack([x, y])


def create_material_grid(width: float, height: float, cell_mm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(0.0, width + cell_mm, cell_mm)
    y = np.arange(0.0, height + cell_mm, cell_mm)
    xx, yy = np.meshgrid(x, y)
    remaining = np.ones_like(xx, dtype=np.uint8)
    return xx, yy, remaining


def build_disk_offsets(cell_size: float, tool_radius: float) -> tuple[np.ndarray, np.ndarray]:
    r_cells = int(np.ceil(tool_radius / max(cell_size, 1e-12)))
    offsets = np.arange(-r_cells, r_cells + 1, dtype=int)
    ox, oy = np.meshgrid(offsets, offsets)
    mask = (ox * cell_size) ** 2 + (oy * cell_size) ** 2 <= tool_radius**2
    return ox[mask], oy[mask]


def center_to_index(cx: float, cy: float, x0: float, y0: float, cell_x: float, cell_y: float) -> tuple[int, int]:
    ix = int(np.rint((cx - x0) / max(cell_x, 1e-12)))
    iy = int(np.rint((cy - y0) / max(cell_y, 1e-12)))
    return ix, iy


def carve_2d(remaining: np.ndarray, ix: int, iy: int, ox: np.ndarray, oy: np.ndarray) -> None:
    xs = ix + ox
    ys = iy + oy
    valid = (xs >= 0) & (xs < remaining.shape[1]) & (ys >= 0) & (ys < remaining.shape[0])
    remaining[ys[valid], xs[valid]] = 0


def carve_3d(z_top: np.ndarray, ix: int, iy: int, ox: np.ndarray, oy: np.ndarray, depth: float) -> None:
    xs = ix + ox
    ys = iy + oy
    valid = (xs >= 0) & (xs < z_top.shape[1]) & (ys >= 0) & (ys < z_top.shape[0])
    ys_valid = ys[valid]
    xs_valid = xs[valid]
    z_top[ys_valid, xs_valid] = np.minimum(z_top[ys_valid, xs_valid], -depth)


def draw_tool(ax, cx: float, cy: float, tool_radius: float, top_z: float, bottom_z: float) -> None:
    # Smooth parametric cylinder body.
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    z = np.linspace(bottom_z, top_z, 24)
    tt, zz = np.meshgrid(theta, z)
    xx = cx + tool_radius * np.cos(tt)
    yy = cy + tool_radius * np.sin(tt)

    # Manual lighting around the circumference for a cleaner cylindrical look.
    light_phase = tt - np.deg2rad(30.0)
    light = 0.62 + 0.38 * (0.5 + 0.5 * np.cos(light_phase))
    side_colors = np.zeros(xx.shape + (4,), dtype=float)
    side_colors[..., 0] = 0.58 + 0.37 * light
    side_colors[..., 1] = 0.07 + 0.10 * light
    side_colors[..., 2] = 0.07 + 0.10 * light
    side_colors[..., 3] = 1.0

    tool_surface = ax.plot_surface(
        xx,
        yy,
        zz,
        facecolors=side_colors,
        edgecolor="none",
        linewidth=0,
        shade=False,
        antialiased=True,
        zorder=80,
    )
    if hasattr(tool_surface, "set_zsort"):
        tool_surface.set_zsort("max")
    if hasattr(tool_surface, "set_sort_zpos"):
        tool_surface.set_sort_zpos(1e6)

    # Optional clean top cap: single smooth disk (no circles/markers/mesh clutter).
    r = np.linspace(0.0, tool_radius, 22)
    rr, tt_top = np.meshgrid(r, theta)
    x_top = cx + rr * np.cos(tt_top)
    y_top = cy + rr * np.sin(tt_top)
    z_top_disk = np.full_like(x_top, top_z)
    top_light = 0.72 + 0.28 * (0.5 + 0.5 * np.cos(tt_top - np.deg2rad(30.0)))
    top_colors = np.zeros(x_top.shape + (4,), dtype=float)
    top_colors[..., 0] = 0.66 + 0.29 * top_light
    top_colors[..., 1] = 0.10 + 0.08 * top_light
    top_colors[..., 2] = 0.10 + 0.08 * top_light
    top_colors[..., 3] = 1.0

    top_surface = ax.plot_surface(
        x_top,
        y_top,
        z_top_disk,
        facecolors=top_colors,
        edgecolor="none",
        linewidth=0,
        shade=False,
        antialiased=True,
        zorder=81,
    )
    if hasattr(top_surface, "set_zsort"):
        top_surface.set_zsort("max")
    if hasattr(top_surface, "set_sort_zpos"):
        top_surface.set_sort_zpos(1e6 + 0.5)

    # Thin outline only, to keep the tool defined without mesh noise.
    theta_outline = np.linspace(0.0, 2.0 * np.pi, 180)
    ax.plot(
        cx + tool_radius * np.cos(theta_outline),
        cy + tool_radius * np.sin(theta_outline),
        np.full(theta_outline.shape, top_z),
        color="#111111",
        linewidth=0.7,
        zorder=82,
    )


def draw_tool_spec_panel(ax, tool_spec: dict[str, float | int | str]) -> None:
    """Draw a compact static cutter schematic and key tool data for the animation."""
    ax.cla()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f8f8f8")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#9a9a9a")

    # Simple side-view tool model (shank + fluted section).
    shank_x0, shank_x1 = 0.10, 0.26
    flute_x0, flute_x1 = 0.26, 0.38
    body_y0, body_y1 = 0.10, 0.90
    flute_y0, flute_y1 = 0.10, 0.58
    ax.fill([shank_x0, shank_x1, shank_x1, shank_x0], [body_y0, body_y0, body_y1, body_y1], color="#c9ced4")
    ax.fill([flute_x0, flute_x1, flute_x1, flute_x0], [body_y0, body_y0, body_y1, body_y1], color="#b9bfc7")
    ax.fill([flute_x0, flute_x1, flute_x1, flute_x0], [flute_y0, flute_y0, flute_y1, flute_y1], color="#d44a3a")

    # Helical flute hints.
    y_curve = np.linspace(flute_y0 + 0.02, flute_y1 - 0.02, 80)
    for offset in (-0.016, 0.0, 0.016):
        x_curve = flute_x0 + 0.020 + (flute_x1 - flute_x0 - 0.040) * (0.5 + 0.5 * np.sin(9.0 * y_curve + offset * 100.0))
        ax.plot(x_curve, y_curve, color="#8b1d17", linewidth=1.0)

    # Title and tool metadata.
    ax.text(0.44, 0.93, "Tool Model", fontsize=9.4, fontweight="bold", ha="left", va="top", color="#1d1d1d")
    lines = [
        f"Type: {tool_spec['type']}",
        f"Dia: {tool_spec['diameter_mm']:.2f} mm",
        f"Flutes: {int(tool_spec['flutes'])}",
        f"Flute L: {tool_spec['flute_length_mm']:.1f} mm",
        f"Stickout: {tool_spec['stickout_mm']:.1f} mm",
        f"Helix: {tool_spec['helix_deg']:.0f} deg",
        f"Material: {tool_spec['tool_material']}",
    ]
    y_text = 0.84
    for line in lines:
        ax.text(0.44, y_text, line, fontsize=8.3, ha="left", va="top", color="#2a2a2a")
        y_text -= 0.10


def resolve_tool_spec(stepover: float) -> dict[str, float | int | str]:
    """Build complete tool specification from single config + process rule(s)."""
    spec = dict(TOOL_CONFIG)
    # Program rule: tool diameter remains coupled to stepover in this model.
    spec["diameter_mm"] = float(2.0 * stepover)
    return spec


def main() -> None:
    width = 10.0
    height = 7.0
    stepover = 0.8
    tool_spec = resolve_tool_spec(stepover)
    tool_diameter = float(tool_spec["diameter_mm"])
    tool_radius = tool_diameter / 2.0
    pocket_depth = 2.4
    stock_thickness = 4.0
    perturb_amp = 0.18
    perturb_wavelength = 1.4
    samples = 280

    path_base = build_pocket_raster(width=width, height=height, stepover=stepover)
    path_base = resample_polyline(path_base, samples=samples)
    path_pert = apply_sinusoidal_perturbation(path_base, amplitude=perturb_amp, wavelength=perturb_wavelength)

    xx2d_b, yy2d_b, rem2d_b = create_material_grid(width, height, cell_mm=0.07)
    xx2d_p, yy2d_p, rem2d_p = create_material_grid(width, height, cell_mm=0.07)
    cell_x2d = float(xx2d_b[0, 1] - xx2d_b[0, 0])
    cell_y2d = float(yy2d_b[1, 0] - yy2d_b[0, 0])
    x0_2d = float(xx2d_b[0, 0])
    y0_2d = float(yy2d_b[0, 0])
    ox2d, oy2d = build_disk_offsets(min(cell_x2d, cell_y2d), tool_radius)
    cmap = ListedColormap(["#a9a9a9", "#f4d79a"])  # removed, remaining

    x3 = np.linspace(0.0, width, 70)
    y3 = np.linspace(0.0, height, 52)
    xx3, yy3 = np.meshgrid(x3, y3)
    # Precomputed tonal maps to give removed faces a more shape-like appearance.
    cut_tone = 0.42 + 0.24 * (0.5 + 0.5 * np.sin(2.0 * np.pi * xx3 / max(width, 1e-9)) * np.cos(2.0 * np.pi * yy3 / max(height, 1e-9)))
    stock_tone = 0.94 + 0.05 * (0.5 + 0.5 * np.cos(2.0 * np.pi * xx3 / max(width, 1e-9)))
    cell_x3 = float(x3[1] - x3[0])
    cell_y3 = float(y3[1] - y3[0])
    x0_3d = float(x3[0])
    y0_3d = float(y3[0])
    ox3d, oy3d = build_disk_offsets(min(cell_x3, cell_y3), tool_radius)
    z3_b = np.zeros_like(xx3)
    z3_p = np.zeros_like(xx3)

    fig = plt.figure(figsize=(15.0, 10.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.5], hspace=0.14, wspace=0.08)
    ax2d_b = fig.add_subplot(grid[0, 0])
    ax2d_p = fig.add_subplot(grid[0, 1])
    ax3d_b = fig.add_subplot(grid[1, 0], projection="3d")
    ax3d_p = fig.add_subplot(grid[1, 1], projection="3d")
    # Static metadata panel: does not alter the 2x2 axes layout.
    ax_tool_card = fig.add_axes([0.785, 0.735, 0.205, 0.245])
    fig.patch.set_facecolor("white")
    draw_tool_spec_panel(ax_tool_card, tool_spec)

    im_b = ax2d_b.imshow(rem2d_b, origin="lower", extent=[0, width, 0, height], cmap=cmap, vmin=0, vmax=1, alpha=0.9)
    im_p = ax2d_p.imshow(rem2d_p, origin="lower", extent=[0, width, 0, height], cmap=cmap, vmin=0, vmax=1, alpha=0.9)
    line2d_b, = ax2d_b.plot([], [], color="black", linewidth=1.6)
    line2d_p, = ax2d_p.plot([], [], color="#1f77b4", linewidth=1.6)
    tool_color_2d = "#d62728"
    circ_b = Circle((0.0, 0.0), tool_radius, fill=False, color=tool_color_2d, linewidth=1.5)
    circ_p = Circle((0.0, 0.0), tool_radius, fill=False, color=tool_color_2d, linewidth=1.5)
    ax2d_b.add_patch(circ_b)
    ax2d_p.add_patch(circ_p)
    contour_b = [None]
    contour_p = [None]
    contour_stride = 3

    for ax, title in [(ax2d_b, "2D Unperturbed"), (ax2d_p, "2D Perturbed")]:
        ax.set_xlim(-0.3, width + 0.3)
        ax.set_ylim(-0.3, height + 0.3)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(title)

    def draw_3d_panel(ax, z_top: np.ndarray, path: np.ndarray, frame: int, title: str, trace_color: str) -> None:
        ax.cla()
        if hasattr(ax, "computed_zorder"):
            ax.computed_zorder = False
        removed_mask = z_top <= -pocket_depth + 1e-9
        colors = np.zeros(z_top.shape + (4,), dtype=float)
        colors[..., 0] = np.where(removed_mask, cut_tone, 0.96 * stock_tone)
        colors[..., 1] = np.where(removed_mask, cut_tone, 0.83 * stock_tone)
        colors[..., 2] = np.where(removed_mask, cut_tone, 0.60 * stock_tone)
        colors[..., 3] = np.where(removed_mask, 0.45, 0.58)
        stock_surface = ax.plot_surface(
            xx3, yy3, z_top, facecolors=colors, shade=False, linewidth=0, antialiased=False, zorder=1
        )
        if hasattr(stock_surface, "set_zsort"):
            stock_surface.set_zsort("min")
        if hasattr(stock_surface, "set_sort_zpos"):
            stock_surface.set_sort_zpos(-1e6)
        ax.plot(
            path[: frame + 1, 0],
            path[: frame + 1, 1],
            np.full(frame + 1, 0.05),
            color=trace_color,
            linewidth=1.0,
            linestyle="--",
            zorder=40,
        )
        cx, cy = float(path[frame, 0]), float(path[frame, 1])
        # Draw the cutting tool after all potentially obstructing surfaces.
        draw_tool(ax, cx, cy, tool_radius=tool_radius, top_z=0.5, bottom_z=-pocket_depth)
        ax.set_xlim(-0.3, width + 0.3)
        ax.set_ylim(-0.3, height + 0.3)
        ax.set_zlim(-stock_thickness, 1.0)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        ax.view_init(elev=28, azim=-58)
        ax.set_box_aspect((width, height, stock_thickness * 0.9))

    def update(frame: int):
        bx, by = float(path_base[frame, 0]), float(path_base[frame, 1])
        px, py = float(path_pert[frame, 0]), float(path_pert[frame, 1])

        bix, biy = center_to_index(bx, by, x0_2d, y0_2d, cell_x2d, cell_y2d)
        pix, piy = center_to_index(px, py, x0_2d, y0_2d, cell_x2d, cell_y2d)
        carve_2d(rem2d_b, bix, biy, ox2d, oy2d)
        carve_2d(rem2d_p, pix, piy, ox2d, oy2d)

        b3x, b3y = center_to_index(bx, by, x0_3d, y0_3d, cell_x3, cell_y3)
        p3x, p3y = center_to_index(px, py, x0_3d, y0_3d, cell_x3, cell_y3)
        carve_3d(z3_b, b3x, b3y, ox3d, oy3d, pocket_depth)
        carve_3d(z3_p, p3x, p3y, ox3d, oy3d, pocket_depth)

        im_b.set_data(rem2d_b)
        im_p.set_data(rem2d_p)
        if frame % contour_stride == 0 or frame == len(path_base) - 1:
            if contour_b[0] is not None:
                contour_b[0].remove()
            if contour_p[0] is not None:
                contour_p[0].remove()
            contour_b[0] = ax2d_b.contour(
                xx2d_b, yy2d_b, (rem2d_b == 0).astype(float), levels=[0.5], colors="#5a5a5a", linewidths=0.8
            )
            contour_p[0] = ax2d_p.contour(
                xx2d_p, yy2d_p, (rem2d_p == 0).astype(float), levels=[0.5], colors="#5a5a5a", linewidths=0.8
            )
        line2d_b.set_data(path_base[: frame + 1, 0], path_base[: frame + 1, 1])
        line2d_p.set_data(path_pert[: frame + 1, 0], path_pert[: frame + 1, 1])
        circ_b.center = (bx, by)
        circ_p.center = (px, py)

        draw_3d_panel(ax3d_b, z3_b, path_base, frame, "3D Unperturbed", "#4a4a4a")
        draw_3d_panel(ax3d_p, z3_p, path_pert, frame, "3D Perturbed", "#4a4a4a")
        return im_b, im_p, line2d_b, line2d_p, circ_b, circ_p

    anim = FuncAnimation(fig, update, frames=len(path_base), interval=90, repeat=False, blit=False)
    anim.save(OUTPUT_GIF, writer=PillowWriter(fps=11))
    plt.close(fig)
    print(f"Created {OUTPUT_GIF}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import cadquery as cq
    HAS_CADQUERY = True
except Exception:
    cq = None
    HAS_CADQUERY = False


@dataclass(frozen=True)
class ToolSpec:
    tool_type: str
    diameter_mm: float
    flutes: int
    flute_length_mm: float
    stickout_mm: float
    helix_deg: float
    tool_material: str


def build_endmill_solid(spec: ToolSpec):
    """Build a visually convincing flat-end mill solid using helical flute cuts.

    Notes:
    - This is a practical CAD approximation (engineering-visual grade), not a manufacturing-true cutter grind.
    - Flutes are modeled by twist-extruded subtractive slots.
    """
    radius = 0.5 * spec.diameter_mm

    # Main cylindrical body.
    body = cq.Workplane("XY").circle(radius).extrude(spec.stickout_mm)

    # Flute twist from helix angle and circumference travel.
    # tan(helix) = circumferential travel / axial travel
    circumferential_travel = spec.flute_length_mm * np.tan(np.deg2rad(spec.helix_deg))
    turns = circumferential_travel / max(2.0 * np.pi * radius, 1e-9)
    twist_deg = float(turns * 360.0)

    flute_profile_w = max(0.18 * spec.diameter_mm, 0.12)
    flute_profile_d = max(0.58 * spec.diameter_mm, 0.20)
    flute_offset = 0.26 * spec.diameter_mm

    for i in range(spec.flutes):
        angle = i * (360.0 / spec.flutes)

        flute_cut = (
            cq.Workplane("XY")
            .center(flute_offset, 0.0)
            .rect(flute_profile_w, flute_profile_d)
            .twistExtrude(spec.flute_length_mm, angleDegrees=twist_deg, combine=False)
            .rotate((0, 0, 0), (0, 0, 1), angle)
        )
        body = body.cut(flute_cut)

    # Keep a flat-end appearance with slight edge break at tip.
    try:
        body = body.faces("<Z").chamfer(min(0.02 * spec.diameter_mm, 0.08))
    except Exception:
        pass

    return body


def render_solid_to_png(solid, png_path: Path, elev: float = 24.0, azim: float = -52.0) -> None:
    """Render a clean shaded image from the CAD solid tessellation."""
    shape = solid.val()
    verts, tri_idx = shape.tessellate(0.04)

    v = np.array([[p.x, p.y, p.z] for p in verts], dtype=float)
    f = np.asarray(tri_idx, dtype=int)
    tri = v[f]

    # Face normals for simple physically-plausible shading.
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(n_norm, 1e-12)

    light_dir = np.array([0.45, -0.30, 0.84], dtype=float)
    light_dir = light_dir / np.linalg.norm(light_dir)
    intensity = np.clip(n @ light_dir, 0.0, 1.0)

    base = np.array([0.62, 0.64, 0.67], dtype=float)  # carbide-like neutral metallic gray
    face_rgb = (0.40 + 0.60 * intensity[:, None]) * base[None, :]
    face_rgba = np.column_stack([face_rgb, np.ones(len(face_rgb), dtype=float)])

    fig = plt.figure(figsize=(4.0, 4.0), dpi=260)
    ax = fig.add_subplot(111, projection="3d")

    poly = Poly3DCollection(tri, linewidths=0.0, antialiaseds=True)
    poly.set_facecolor(face_rgba)
    poly.set_edgecolor((0, 0, 0, 0))
    ax.add_collection3d(poly)

    mins = v.min(axis=0)
    maxs = v.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins) * 0.56)
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(mins[2] - 0.03 * (maxs[2] - mins[2]), maxs[2] + 0.05 * (maxs[2] - mins[2]))

    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.0, 1.0, 2.4))

    # Clean render: no axes, panes, or frame clutter.
    ax.set_axis_off()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(png_path, transparent=True)
    plt.close(fig)


def build_procedural_tool_mesh(spec: ToolSpec) -> np.ndarray:
    """Build a triangulated 3D tool mesh without CAD dependencies.

    The geometry is a true 3D solid-like surface with helical flute modulation
    over the flute section and full cylindrical shank above it.
    """
    radius = 0.5 * spec.diameter_mm
    n_theta = 104
    n_z = 240
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    z = np.linspace(0.0, spec.stickout_mm, n_z)
    th_grid, z_grid = np.meshgrid(theta, z)

    r = np.full_like(th_grid, radius)
    flute_mask = z_grid <= spec.flute_length_mm

    if np.any(flute_mask) and spec.flutes > 0:
        helix_phase = (z_grid / max(radius, 1e-8)) * np.tan(np.deg2rad(spec.helix_deg))
        groove_strength = np.zeros_like(r)
        for i in range(spec.flutes):
            phase_i = i * (2.0 * np.pi / spec.flutes) + helix_phase
            groove_strength = np.maximum(groove_strength, np.exp(4.2 * (np.cos(th_grid - phase_i) - 1.0)))
        depth = np.clip(0.18 * spec.diameter_mm, 0.08, 0.55 * radius)
        r[flute_mask] = np.maximum(0.28 * radius, radius - depth * groove_strength[flute_mask])

    x = r * np.cos(th_grid)
    y = r * np.sin(th_grid)

    triangles: list[np.ndarray] = []
    for j in range(n_z - 1):
        for i in range(n_theta):
            i2 = (i + 1) % n_theta
            p00 = np.array([x[j, i], y[j, i], z[j]])
            p10 = np.array([x[j, i2], y[j, i2], z[j]])
            p01 = np.array([x[j + 1, i], y[j + 1, i], z[j + 1]])
            p11 = np.array([x[j + 1, i2], y[j + 1, i2], z[j + 1]])
            triangles.append(np.stack([p00, p10, p11], axis=0))
            triangles.append(np.stack([p00, p11, p01], axis=0))

    tip_center = np.array([0.0, 0.0, 0.0])
    top_center = np.array([0.0, 0.0, spec.stickout_mm])
    for i in range(n_theta):
        i2 = (i + 1) % n_theta
        tip_a = np.array([x[0, i], y[0, i], z[0]])
        tip_b = np.array([x[0, i2], y[0, i2], z[0]])
        top_a = np.array([x[-1, i], y[-1, i], z[-1]])
        top_b = np.array([x[-1, i2], y[-1, i2], z[-1]])
        triangles.append(np.stack([tip_center, tip_b, tip_a], axis=0))
        triangles.append(np.stack([top_center, top_a, top_b], axis=0))

    return np.stack(triangles, axis=0)


def render_triangles_to_png(tri: np.ndarray, png_path: Path, elev: float = 24.0, azim: float = -52.0) -> None:
    """Render pre-triangulated mesh to a clean transparent PNG."""
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(n_norm, 1e-12)

    light_dir = np.array([0.45, -0.30, 0.84], dtype=float)
    light_dir = light_dir / np.linalg.norm(light_dir)
    intensity = np.clip(n @ light_dir, 0.0, 1.0)

    base = np.array([0.62, 0.64, 0.67], dtype=float)
    face_rgb = (0.40 + 0.60 * intensity[:, None]) * base[None, :]
    face_rgba = np.column_stack([face_rgb, np.ones(len(face_rgb), dtype=float)])

    fig = plt.figure(figsize=(4.0, 4.0), dpi=260)
    ax = fig.add_subplot(111, projection="3d")

    poly = Poly3DCollection(tri, linewidths=0.0, antialiaseds=True)
    poly.set_facecolor(face_rgba)
    poly.set_edgecolor((0, 0, 0, 0))
    ax.add_collection3d(poly)

    v = tri.reshape(-1, 3)
    mins = v.min(axis=0)
    maxs = v.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins) * 0.56)
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(mins[2] - 0.03 * (maxs[2] - mins[2]), maxs[2] + 0.05 * (maxs[2] - mins[2]))

    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.0, 1.0, 2.4))
    ax.set_axis_off()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(png_path, transparent=True)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and render a parametric end-mill CAD model.")
    parser.add_argument("--diameter-mm", type=float, required=True)
    parser.add_argument("--flutes", type=int, required=True)
    parser.add_argument("--flute-length-mm", type=float, required=True)
    parser.add_argument("--stickout-mm", type=float, required=True)
    parser.add_argument("--helix-deg", type=float, required=True)
    parser.add_argument("--tool-type", type=str, default="Flat End Mill")
    parser.add_argument("--tool-material", type=str, default="Carbide")
    parser.add_argument("--png", type=str, default="tool_model_render.png")
    parser.add_argument("--stl", type=str, default="tool_model_render.stl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = ToolSpec(
        tool_type=args.tool_type,
        diameter_mm=float(args.diameter_mm),
        flutes=int(args.flutes),
        flute_length_mm=float(args.flute_length_mm),
        stickout_mm=float(args.stickout_mm),
        helix_deg=float(args.helix_deg),
        tool_material=args.tool_material,
    )

    stl_path = Path(args.stl)
    png_path = Path(args.png)
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    if HAS_CADQUERY:
        try:
            solid = build_endmill_solid(spec)
            # CadQuery exporter expects string-like filenames.
            cq.exporters.export(solid, str(stl_path))
            render_solid_to_png(solid, png_path)
            print(f"Created STL: {stl_path}")
            print(f"Created PNG: {png_path}")
            return
        except Exception as exc:
            print(f"CadQuery render failed ({exc}); falling back to procedural render.")

    else:
        tri = build_procedural_tool_mesh(spec)
        render_triangles_to_png(tri, png_path)
        print("CadQuery/OCP backend unavailable; generated procedural 3D tool render only.")
        print(f"Created PNG: {png_path}")
        return

    tri = build_procedural_tool_mesh(spec)
    render_triangles_to_png(tri, png_path)
    print("Generated procedural 3D tool render fallback.")
    print(f"Created PNG: {png_path}")


if __name__ == "__main__":
    main()

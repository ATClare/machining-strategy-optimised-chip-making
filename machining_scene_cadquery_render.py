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
    HAS_LOCAL_CADQUERY = True
except Exception:
    cq = None
    HAS_LOCAL_CADQUERY = False

from tool_model_cadquery_render import ToolSpec, build_endmill_solid, build_procedural_tool_mesh


@dataclass(frozen=True)
class SceneSpec:
    stock_x_mm: float = 120.0
    stock_y_mm: float = 82.0
    stock_z_mm: float = 24.0
    boss_diameter_mm: float = 52.0
    boss_height_mm: float = 14.0
    tool_diameter_mm: float = 14.0
    tool_flutes: int = 4
    tool_flute_len_mm: float = 32.0
    tool_stickout_mm: float = 78.0
    tool_helix_deg: float = 35.0
    tool_engagement_mm: float = 0.15


def _tessellate_solid(solid, linear_tol_mm: float = 0.012, angular_tol_rad: float = 0.06) -> np.ndarray:
    shape = solid.val() if hasattr(solid, "val") else solid
    # Tight tessellation tolerance strongly reduces visible faceting.
    try:
        verts, tri_idx = shape.tessellate(float(linear_tol_mm), float(angular_tol_rad))
    except TypeError:
        verts, tri_idx = shape.tessellate(float(linear_tol_mm))

    v = np.array([[p.x, p.y, p.z] for p in verts], dtype=float)
    f = np.asarray(tri_idx, dtype=int)
    return v[f]


def _shade_triangles(tri: np.ndarray, base_rgb: tuple[float, float, float]) -> np.ndarray:
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(n_norm, 1e-12)

    key = np.array([0.55, -0.36, 0.75], dtype=float)
    fill = np.array([-0.18, 0.78, 0.60], dtype=float)
    key = key / np.linalg.norm(key)
    fill = fill / np.linalg.norm(fill)

    k = np.clip(n @ key, 0.0, 1.0)
    f = np.clip(n @ fill, 0.0, 1.0)
    intensity = np.clip(0.70 * k + 0.30 * f, 0.0, 1.0)

    base = np.array(base_rgb, dtype=float)
    rgb = np.clip((0.60 + 0.40 * intensity[:, None]) * base[None, :], 0.0, 1.0)
    return np.column_stack([rgb, np.ones(len(rgb), dtype=float)])


def _translate_tri(tri: np.ndarray, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> np.ndarray:
    out = np.array(tri, copy=True)
    out[:, :, 0] += dx
    out[:, :, 1] += dy
    out[:, :, 2] += dz
    return out


def _box_triangles(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> np.ndarray:
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )
    faces = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]
    return np.stack([corners[list(face)] for face in faces], axis=0)


def _cylinder_triangles(
    radius: float,
    z0: float,
    z1: float,
    n_theta: int = 360,
    include_bottom_cap: bool = True,
    include_top_cap: bool = True,
) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    lower = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.full_like(theta, z0)])
    upper = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.full_like(theta, z1)])

    tris: list[np.ndarray] = []
    for i in range(n_theta):
        i2 = (i + 1) % n_theta
        tris.append(np.stack([lower[i], lower[i2], upper[i2]], axis=0))
        tris.append(np.stack([lower[i], upper[i2], upper[i]], axis=0))

    if include_bottom_cap:
        lower_center = np.array([0.0, 0.0, z0], dtype=float)
        for i in range(n_theta):
            i2 = (i + 1) % n_theta
            tris.append(np.stack([lower_center, lower[i2], lower[i]], axis=0))

    if include_top_cap:
        upper_center = np.array([0.0, 0.0, z1], dtype=float)
        for i in range(n_theta):
            i2 = (i + 1) % n_theta
            tris.append(np.stack([upper_center, upper[i], upper[i2]], axis=0))
    return np.stack(tris, axis=0)


def _frustum_triangles(r0: float, r1: float, z0: float, z1: float, n_theta: int = 360) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    lower = np.column_stack([r0 * np.cos(theta), r0 * np.sin(theta), np.full_like(theta, z0)])
    upper = np.column_stack([r1 * np.cos(theta), r1 * np.sin(theta), np.full_like(theta, z1)])

    tris: list[np.ndarray] = []
    for i in range(n_theta):
        i2 = (i + 1) % n_theta
        tris.append(np.stack([lower[i], lower[i2], upper[i2]], axis=0))
        tris.append(np.stack([lower[i], upper[i2], upper[i]], axis=0))
    return np.stack(tris, axis=0)


def _build_scene_components_cadquery(spec: SceneSpec) -> list[tuple[np.ndarray, tuple[float, float, float]]]:
    stock_top = spec.stock_z_mm
    boss_radius = 0.5 * spec.boss_diameter_mm
    boss_top = stock_top + spec.boss_height_mm

    # Slight XY offset makes facing engagement unmistakable in the render.
    tool_x = -0.48 * boss_radius
    tool_y = 0.34 * boss_radius
    tool_tip_z = boss_top - spec.tool_engagement_mm

    stock = (
        cq.Workplane("XY")
        .box(spec.stock_x_mm, spec.stock_y_mm, spec.stock_z_mm)
        .translate((0.0, 0.0, 0.5 * spec.stock_z_mm))
    )
    boss = (
        cq.Workplane("XY")
        .workplane(offset=stock_top)
        .circle(boss_radius)
        .extrude(spec.boss_height_mm)
    )

    # A shallow skim patch highlights the current cut pass on the boss top face.
    machined_patch = (
        cq.Workplane("XY")
        .center(tool_x, tool_y)
        .workplane(offset=boss_top - 0.10)
        .circle(0.40 * spec.tool_diameter_mm)
        .extrude(0.11)
    )

    tool_spec = ToolSpec(
        tool_type="Flat End Mill",
        diameter_mm=spec.tool_diameter_mm,
        flutes=spec.tool_flutes,
        flute_length_mm=spec.tool_flute_len_mm,
        stickout_mm=spec.tool_stickout_mm,
        helix_deg=spec.tool_helix_deg,
        tool_material="Carbide",
    )
    endmill = build_endmill_solid(tool_spec).translate((tool_x, tool_y, tool_tip_z))

    collet_radius = 0.46 * spec.tool_diameter_mm
    collet_len = 18.0
    collet_z0 = tool_tip_z + spec.tool_stickout_mm - 11.0
    collet_z1 = collet_z0 + collet_len

    holder_small_r = 16.0
    holder_big_r = 22.0
    holder_taper_len = 38.0
    holder_body_len = 24.0
    holder_z1 = collet_z1 + holder_taper_len
    holder_body_z1 = holder_z1 + holder_body_len

    flange_radius = 33.0
    flange_thickness = 10.5
    flange_z1 = holder_body_z1 + flange_thickness

    spindle_nose_small_r = 36.0
    spindle_nose_big_r = 43.0
    spindle_nose_len = 24.0
    spindle_nose_z1 = flange_z1 + spindle_nose_len
    spindle_body_r = 44.0
    spindle_body_len = 28.0

    collet = (
        cq.Workplane("XY")
        .workplane(offset=collet_z0)
        .circle(collet_radius)
        .extrude(collet_len)
    )
    collet_nut = (
        cq.Workplane("XY")
        .workplane(offset=collet_z0 - 8.0)
        .circle(collet_radius + 3.0)
        .extrude(9.0)
    )

    holder_taper = (
        cq.Workplane("XY")
        .workplane(offset=collet_z1)
        .circle(holder_small_r)
        .workplane(offset=holder_taper_len)
        .circle(holder_big_r)
        .loft(combine=True)
    )
    holder_body = (
        cq.Workplane("XY")
        .workplane(offset=holder_z1)
        .circle(holder_big_r)
        .extrude(holder_body_len)
    )
    holder_flange = (
        cq.Workplane("XY")
        .workplane(offset=holder_body_z1)
        .circle(flange_radius)
        .extrude(flange_thickness)
    )

    spindle_nose = (
        cq.Workplane("XY")
        .workplane(offset=flange_z1)
        .circle(spindle_nose_small_r)
        .workplane(offset=spindle_nose_len)
        .circle(spindle_nose_big_r)
        .loft(combine=True)
    )
    spindle_body = (
        cq.Workplane("XY")
        .workplane(offset=spindle_nose_z1)
        .circle(spindle_body_r)
        .extrude(spindle_body_len)
    )

    holder_assembly = collet.union(collet_nut).union(holder_taper).union(holder_body).union(holder_flange)

    return [
        (_tessellate_solid(stock), (0.69, 0.73, 0.78)),
        (_tessellate_solid(boss), (0.62, 0.75, 0.90)),
        (_tessellate_solid(machined_patch), (0.90, 0.92, 0.95)),
        (_tessellate_solid(spindle_body), (0.35, 0.37, 0.40)),
        (_tessellate_solid(spindle_nose), (0.30, 0.32, 0.35)),
        (_tessellate_solid(holder_assembly), (0.19, 0.20, 0.22)),
        (_tessellate_solid(endmill), (0.70, 0.72, 0.75)),
    ]


def _build_scene_components_procedural(spec: SceneSpec) -> list[tuple[np.ndarray, tuple[float, float, float]]]:
    stock_top = spec.stock_z_mm
    boss_radius = 0.5 * spec.boss_diameter_mm
    boss_top = stock_top + spec.boss_height_mm

    tool_x = -0.48 * boss_radius
    tool_y = 0.34 * boss_radius
    tool_tip_z = boss_top - spec.tool_engagement_mm

    tool_spec = ToolSpec(
        tool_type="Flat End Mill",
        diameter_mm=spec.tool_diameter_mm,
        flutes=spec.tool_flutes,
        flute_length_mm=spec.tool_flute_len_mm,
        stickout_mm=spec.tool_stickout_mm,
        helix_deg=spec.tool_helix_deg,
        tool_material="Carbide",
    )
    tool_tri = build_procedural_tool_mesh(tool_spec)
    tool_tri = _translate_tri(tool_tri, dx=tool_x, dy=tool_y, dz=tool_tip_z)

    stock_tri = _box_triangles(
        -0.5 * spec.stock_x_mm,
        0.5 * spec.stock_x_mm,
        -0.5 * spec.stock_y_mm,
        0.5 * spec.stock_y_mm,
        0.0,
        spec.stock_z_mm,
    )
    boss_tri = _cylinder_triangles(
        boss_radius,
        stock_top,
        boss_top,
        n_theta=260,
        include_bottom_cap=False,
        include_top_cap=True,
    )
    patch_tri = _translate_tri(
        _cylinder_triangles(0.40 * spec.tool_diameter_mm, boss_top - 0.10, boss_top + 0.01, n_theta=220),
        dx=tool_x,
        dy=tool_y,
    )

    collet_radius = 0.46 * spec.tool_diameter_mm
    collet_len = 18.0
    collet_z0 = tool_tip_z + spec.tool_stickout_mm - 11.0
    collet_z1 = collet_z0 + collet_len

    holder_small_r = 16.0
    holder_big_r = 22.0
    holder_taper_len = 38.0
    holder_body_len = 24.0
    holder_z1 = collet_z1 + holder_taper_len
    holder_body_z1 = holder_z1 + holder_body_len

    flange_radius = 33.0
    flange_thickness = 10.5
    flange_z1 = holder_body_z1 + flange_thickness

    spindle_nose_small_r = 36.0
    spindle_nose_big_r = 43.0
    spindle_nose_len = 24.0
    spindle_nose_z1 = flange_z1 + spindle_nose_len
    spindle_body_r = 44.0
    spindle_body_len = 28.0

    collet_tri = _translate_tri(_cylinder_triangles(collet_radius, collet_z0, collet_z1, n_theta=240), dx=tool_x, dy=tool_y)
    collet_nut_tri = _translate_tri(
        _cylinder_triangles(collet_radius + 3.0, collet_z0 - 8.0, collet_z0 + 1.0, n_theta=240),
        dx=tool_x,
        dy=tool_y,
    )
    holder_taper_tri = _translate_tri(
        _frustum_triangles(holder_small_r, holder_big_r, collet_z1, holder_z1, n_theta=240),
        dx=tool_x,
        dy=tool_y,
    )
    holder_body_tri = _translate_tri(
        _cylinder_triangles(holder_big_r, holder_z1, holder_body_z1, n_theta=240),
        dx=tool_x,
        dy=tool_y,
    )
    holder_flange_tri = _translate_tri(
        _cylinder_triangles(flange_radius, holder_body_z1, flange_z1, n_theta=240),
        dx=tool_x,
        dy=tool_y,
    )
    spindle_nose_tri = _translate_tri(
        _frustum_triangles(spindle_nose_small_r, spindle_nose_big_r, flange_z1, spindle_nose_z1, n_theta=240),
        dx=tool_x,
        dy=tool_y,
    )
    spindle_body_tri = _translate_tri(
        _cylinder_triangles(spindle_body_r, spindle_nose_z1, spindle_nose_z1 + spindle_body_len, n_theta=260),
        dx=tool_x,
        dy=tool_y,
    )

    holder_tri = np.concatenate([collet_tri, collet_nut_tri, holder_taper_tri, holder_body_tri, holder_flange_tri], axis=0)
    return [
        (stock_tri, (0.69, 0.73, 0.78)),
        (boss_tri, (0.62, 0.75, 0.90)),
        (patch_tri, (0.90, 0.92, 0.95)),
        (spindle_body_tri, (0.35, 0.37, 0.40)),
        (spindle_nose_tri, (0.30, 0.32, 0.35)),
        (holder_tri, (0.19, 0.20, 0.22)),
        (tool_tri, (0.70, 0.72, 0.75)),
    ]


def build_scene_components(spec: SceneSpec) -> list[tuple[np.ndarray, tuple[float, float, float]]]:
    if HAS_LOCAL_CADQUERY and cq is not None:
        try:
            return _build_scene_components_cadquery(spec)
        except Exception:
            pass
    return _build_scene_components_procedural(spec)


def render_scene_png(
    components: list[tuple[np.ndarray, tuple[float, float, float]]],
    out_png: Path,
    fig_w_in: float = 5.3,
    fig_h_in: float = 4.4,
    dpi: int = 300,
) -> None:
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    all_tri: list[np.ndarray] = []
    all_fc: list[np.ndarray] = []
    for tri, base_rgb in components:
        all_tri.append(tri)
        all_fc.append(_shade_triangles(tri, base_rgb))

    tri_all = np.concatenate(all_tri, axis=0)
    fc_all = np.concatenate(all_fc, axis=0)

    poly = Poly3DCollection(tri_all, linewidths=0.0, antialiaseds=True)
    poly.set_facecolor(fc_all)
    poly.set_edgecolor((0.0, 0.0, 0.0, 0.0))
    ax.add_collection3d(poly)

    points = tri_all.reshape(-1, 3)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    boss_top = SceneSpec().stock_z_mm + SceneSpec().boss_height_mm
    focus_center = np.array([center[0] - 5.0, center[1] + 3.0, boss_top + 35.0], dtype=float)
    span = float(np.max(maxs - mins) * 0.47)

    ax.set_xlim(focus_center[0] - span, focus_center[0] + span)
    ax.set_ylim(focus_center[1] - 0.84 * span, focus_center[1] + 0.84 * span)
    ax.set_zlim(0.0, focus_center[2] + 0.62 * span)

    # Camera chosen to make boss geometry and light facing engagement obvious.
    ax.view_init(elev=16.5, azim=-118.0)
    ax.set_box_aspect((1.36, 1.0, 0.86))
    ax.set_axis_off()

    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, transparent=True)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and render a clean CAD-style machining scene PNG.")
    parser.add_argument("--png", type=str, default="machining_scene_clean.png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--width-in", type=float, default=5.3)
    parser.add_argument("--height-in", type=float, default=4.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = SceneSpec()
    components = build_scene_components(spec)
    out_png = Path(args.png)
    render_scene_png(
        components,
        out_png=out_png,
        fig_w_in=float(args.width_in),
        fig_h_in=float(args.height_in),
        dpi=int(args.dpi),
    )
    print(f"Created machining scene render: {out_png}")


if __name__ == "__main__":
    main()

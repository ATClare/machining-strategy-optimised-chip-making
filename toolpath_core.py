from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class PocketSpec:
    width: float = 50.0
    height: float = 50.0
    stepover: float = 5.0
    samples: int = 520
    perturb_amplitude: float = 0.18
    perturb_wavelength: float = 1.4
    pocket_depth: float = 2.4


def build_pocket_raster(width: float, height: float, stepover: float) -> np.ndarray:
    y_levels = np.arange(0.0, height + 1e-9, stepover)
    if y_levels[-1] < height:
        y_levels = np.append(y_levels, height)

    points: list[tuple[float, float]] = []
    for i, y in enumerate(y_levels):
        if i % 2 == 0:
            points.extend([(0.0, y), (width, y)])
        else:
            points.extend([(width, y), (0.0, y)])
    return np.asarray(points, dtype=float)


def resample_polyline(path: np.ndarray, samples: int) -> tuple[np.ndarray, np.ndarray]:
    deltas = np.diff(path, axis=0)
    seg_lengths = np.sqrt((deltas**2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cumulative[-1]

    s_targets = np.linspace(0.0, total_length, samples)
    x_new = np.interp(s_targets, cumulative, path[:, 0])
    y_new = np.interp(s_targets, cumulative, path[:, 1])
    return np.column_stack([x_new, y_new]), s_targets


def apply_sinusoidal_perturbation(path: np.ndarray, s_values: np.ndarray, amplitude: float, wavelength: float) -> np.ndarray:
    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])
    tangent_norm = np.sqrt(dx**2 + dy**2) + 1e-12
    tx = dx / tangent_norm
    ty = dy / tangent_norm

    nx = -ty
    ny = tx

    signal = amplitude * np.sin(2.0 * np.pi * s_values / wavelength)
    x_perturbed = path[:, 0] + signal * nx
    y_perturbed = path[:, 1] + signal * ny
    return np.column_stack([x_perturbed, y_perturbed])


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


def carve_height_field(z_top: np.ndarray, ix: int, iy: int, ox: np.ndarray, oy: np.ndarray, depth: float) -> None:
    xs = ix + ox
    ys = iy + oy
    valid = (xs >= 0) & (xs < z_top.shape[1]) & (ys >= 0) & (ys < z_top.shape[0])
    z_top[ys[valid], xs[valid]] = np.minimum(z_top[ys[valid], xs[valid]], -depth)


def pocket_paths(spec: PocketSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coarse = build_pocket_raster(spec.width, spec.height, spec.stepover)
    base_path, s_values = resample_polyline(coarse, samples=spec.samples)
    perturbed = apply_sinusoidal_perturbation(
        base_path,
        s_values=s_values,
        amplitude=spec.perturb_amplitude,
        wavelength=spec.perturb_wavelength,
    )
    return base_path, perturbed, s_values

from __future__ import annotations

from pathlib import Path
import argparse
import csv

import numpy as np
import pandas as pd
import qrcode

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tool_model_cadquery_render import ToolSpec, build_procedural_tool_mesh


MIN_STUDY_DIAMETER_MM = 10.0


def _load_catalog_csv(catalog_csv: Path) -> pd.DataFrame:
    """Load catalog robustly, tolerating unquoted commas in trailing notes fields."""
    with catalog_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        return pd.DataFrame()

    header = rows[0]
    n_cols = len(header)
    fixed_rows: list[list[str]] = []
    for row in rows[1:]:
        if not row:
            continue
        if len(row) > n_cols:
            row = row[: n_cols - 1] + [",".join(row[n_cols - 1 :])]
        elif len(row) < n_cols:
            row = row + [""] * (n_cols - len(row))
        fixed_rows.append(row)

    return pd.DataFrame(fixed_rows, columns=header)


def _shade_faces(tri: np.ndarray) -> np.ndarray:
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(n_norm, 1e-12)

    light_dir = np.array([0.45, -0.30, 0.84], dtype=float)
    light_dir = light_dir / np.linalg.norm(light_dir)
    intensity = np.clip(n @ light_dir, 0.0, 1.0)

    base = np.array([0.62, 0.64, 0.67], dtype=float)
    face_rgb = (0.40 + 0.60 * intensity[:, None]) * base[None, :]
    return np.column_stack([face_rgb, np.ones(len(face_rgb), dtype=float)])


def _to_float(row: pd.Series, key: str, default: float) -> float:
    val = row.get(key, default)
    return float(default if pd.isna(val) else val)


def _to_int(row: pd.Series, key: str, default: int) -> int:
    val = row.get(key, default)
    if pd.isna(val):
        return default
    return int(val)


def _tool_spec_from_row(row: pd.Series) -> ToolSpec:
    flute_length = _to_float(row, "flute_length_mm", 12.0)
    overall_length = _to_float(row, "overall_length_mm", flute_length + 35.0)

    return ToolSpec(
        tool_type=str(row.get("tool_type", "Square End Mill")),
        diameter_mm=_to_float(row, "diameter_mm", 6.0),
        flutes=_to_int(row, "flutes", 3),
        flute_length_mm=flute_length,
        stickout_mm=overall_length,
        helix_deg=_to_float(row, "helix_deg", 35.0),
        tool_material=str(row.get("substrate", "Carbide")),
    )


def _build_qr_image(url: str, size_px: int = 180) -> np.ndarray:
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=6,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("L")
    img = img.resize((size_px, size_px))
    return np.asarray(img)


def build_tools_figure(catalog_csv: Path, out_png: Path, min_diameter_mm: float = MIN_STUDY_DIAMETER_MM) -> Path:
    df = _load_catalog_csv(catalog_csv)
    df["diameter_mm"] = pd.to_numeric(df["diameter_mm"], errors="coerce")
    df = df[df["diameter_mm"] >= float(min_diameter_mm)].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No tools found at or above {min_diameter_mm:.1f} mm in catalog.")

    specs: list[ToolSpec] = [_tool_spec_from_row(row) for _, row in df.iterrows()]

    max_radius = max(0.5 * s.diameter_mm for s in specs)
    max_length = max(s.stickout_mm for s in specs)
    xy_span = max(1.25 * max_radius, 1.0)

    n = len(specs)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))

    fig = plt.figure(figsize=(4.2 * n_cols, 5.0 * n_rows), dpi=220)
    fig.suptitle("Tooling List", fontsize=17, fontweight="bold", y=0.985)

    for idx, (_, row) in enumerate(df.iterrows()):
        spec = specs[idx]
        tri = build_procedural_tool_mesh(spec)
        face_rgba = _shade_faces(tri)

        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")
        poly = Poly3DCollection(tri, linewidths=0.0, antialiaseds=True)
        poly.set_facecolor(face_rgba)
        poly.set_edgecolor((0, 0, 0, 0))
        ax.add_collection3d(poly)

        ax.set_xlim(-xy_span, xy_span)
        ax.set_ylim(-xy_span, xy_span)
        ax.set_zlim(0.0, max_length)
        ax.set_box_aspect((1.0, 1.0, max_length / max(2.0 * xy_span, 1e-9)))
        ax.view_init(elev=19, azim=-50)
        ax.set_axis_off()

        title = str(row.get("tool_name", f"Tool {idx + 1}"))
        ax.set_title(title, fontsize=10, pad=8, fontweight="bold")

        details = [
            f"D: {_to_float(row, 'diameter_mm', spec.diameter_mm):.1f} mm | Z: {_to_int(row, 'flutes', spec.flutes)} | Helix: {_to_float(row, 'helix_deg', spec.helix_deg):.0f} deg",
            f"Flute: {_to_float(row, 'flute_length_mm', spec.flute_length_mm):.1f} mm | OAL: {_to_float(row, 'overall_length_mm', spec.stickout_mm):.1f} mm",
            f"Coating: {row.get('coating', 'n/a')} | Substrate: {row.get('substrate', 'n/a')}",
            f"fz rec: {_to_float(row, 'recommended_fz_min_mm_per_tooth', 0.0):.3f}-{_to_float(row, 'recommended_fz_max_mm_per_tooth', 0.0):.3f} mm/tooth",
        ]
        for line_i, line in enumerate(details):
            ax.text2D(0.5, -0.13 - 0.065 * line_i, line, transform=ax.transAxes, ha="center", va="top", fontsize=8.2)

        product_url = str(row.get("product_url", "")).strip()
        if product_url:
            qr_img = _build_qr_image(product_url, size_px=170)
            qr_ax = ax.inset_axes([0.73, 0.00, 0.24, 0.24], transform=ax.transAxes)
            qr_ax.imshow(qr_img, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
            qr_ax.set_xticks([])
            qr_ax.set_yticks([])
            qr_ax.set_facecolor("white")
            for spine in qr_ax.spines.values():
                spine.set_edgecolor("#9ca3af")
                spine.set_linewidth(0.8)
            ax.text2D(0.85, 0.26, "QR", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    total_axes = n_rows * n_cols
    for idx in range(n, total_axes):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")
        ax.set_axis_off()

    plt.subplots_adjust(top=0.94, bottom=0.05, left=0.03, right=0.97, hspace=0.58, wspace=0.18)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, transparent=False)
    plt.close(fig)
    return out_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a to-scale catalog figure of all tools in CSV.")
    parser.add_argument("--csv", type=str, default="cutter_catalog_typical.csv")
    parser.add_argument("--out", type=str, default="tools.png")
    parser.add_argument("--min-diameter-mm", type=float, default=MIN_STUDY_DIAMETER_MM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = build_tools_figure(Path(args.csv), Path(args.out), min_diameter_mm=float(args.min_diameter_mm))
    print(f"Created tools figure: {out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from chip_model_analysis_core import chip_size_map_fixed_tool


def linspace_from_triplet(min_v: float, max_v: float, count: int) -> np.ndarray:
    return np.linspace(float(min_v), float(max_v), int(count))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build a chip-size map for a fixed tool (default: 12 mm, 4 flute) "
            "by sweeping RPM, chip load, axial DOC and radial DOC."
        )
    )
    p.add_argument("--tool-diameter-mm", type=float, default=12.0)
    p.add_argument("--flutes", type=int, default=4)

    p.add_argument("--rpm-min", type=float, default=1000.0)
    p.add_argument("--rpm-max", type=float, default=10000.0)
    p.add_argument("--rpm-count", type=int, default=19)

    p.add_argument("--chipload-min", type=float, default=0.01, help="mm/tooth")
    p.add_argument("--chipload-max", type=float, default=0.20, help="mm/tooth")
    p.add_argument("--chipload-count", type=int, default=20)

    p.add_argument("--axial-doc-min", type=float, default=0.5, help="mm")
    p.add_argument("--axial-doc-max", type=float, default=12.0, help="mm")
    p.add_argument("--axial-doc-count", type=int, default=16)

    p.add_argument("--radial-doc-min", type=float, default=0.2, help="mm")
    p.add_argument("--radial-doc-max", type=float, default=12.0, help="mm")
    p.add_argument("--radial-doc-count", type=int, default=16)

    p.add_argument(
        "--out",
        type=str,
        default="chip_size_map_fixed_tool_12mm_4f.csv",
        help="Output CSV path.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    rpm_values = linspace_from_triplet(args.rpm_min, args.rpm_max, args.rpm_count)
    chipload_values = linspace_from_triplet(args.chipload_min, args.chipload_max, args.chipload_count)
    axial_doc_values = linspace_from_triplet(args.axial_doc_min, args.axial_doc_max, args.axial_doc_count)
    radial_doc_values = linspace_from_triplet(args.radial_doc_min, args.radial_doc_max, args.radial_doc_count)

    df = chip_size_map_fixed_tool(
        rpm_values=rpm_values,
        chipload_values_mm_per_tooth=chipload_values,
        axial_doc_values_mm=axial_doc_values,
        radial_doc_values_mm=radial_doc_values,
        tool_diameter_mm=float(args.tool_diameter_mm),
        flutes=int(args.flutes),
    )

    out_path = Path(args.out)
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path.resolve()}")
    print(f"Rows: {len(df)}")
    print("Columns:", ", ".join(df.columns))


if __name__ == "__main__":
    main()

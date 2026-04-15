# Machining Strategy Optimised for Chip Making

Starter project folder for experimenting with machining strategy ideas, chip evacuation concepts, and related simulation/visualization scripts.

## Files
- `main.py`: Starter entry point.
- `chip_model_analysis_core.py`: Core chip-thickness and process metric equations.
- `chip_size_map_builder.py`: Builds a fixed-tool chip-size map CSV for optimizer input.
- `toolpath_core.py`: Shared pocket/path geometry helpers.
- `viz_style.py`: Common palette and plotting helpers used by animations.
- `toolpath_visualizer.py`: Side-by-side 2D raster vs sinusoid animation with chip metrics.
- `toolpath_2d_merged.py`: Single entry point to build the merged 2D-only GIF.
- `export_notebooks_html.py`: Exports notebook files (`.ipynb`) to `.html`.

## Python Version
- Project target: Python 3.13
- `.python-version` is set to `3.13`.
- VS Code interpreter is pinned to `C:\\Users\\adamc\\AppData\\Local\\Programs\\Python\\Python313\\python.exe` via `.vscode/settings.json`.

## Setup (Windows PowerShell)
```powershell
py -3 -m pip install -U pip
py -3 -m pip install -r requirements.txt
```

## Example Runs
```powershell
# Build 2D merged animation
py -3 toolpath_2d_merged.py

# Build chip-size map for fixed 12 mm, 4 flute tool
py -3 chip_size_map_builder.py

# Export all notebooks to HTML (run this after notebook updates)
py -3 export_notebooks_html.py

# Export a specific notebook to HTML
py -3 export_notebooks_html.py chip_model_unified_flow.ipynb
```

## Chip-Size Map Output
`chip_size_map_builder.py` exports a CSV containing one row per combination of:
- spindle speed (`spindle_rpm`, default sweep up to 10,000 RPM)
- chip load (`chipload_mm_per_tooth`)
- axial depth of cut (`axial_doc_mm`)
- radial depth of cut (`radial_doc_mm`)

Key derived fields include:
- `h_mean_mm`, `h_max_mm`
- `feed_mm_min`
- `mrr_mm3_per_min`
- `chip_contact_length_mm`
- `chip_size_proxy_mm2` (defined as `h_mean_mm * chip_contact_length_mm`)


# Machining Strategy Optimised for Chip Making

Starter project folder for experimenting with machining strategy ideas, chip evacuation concepts, and related simulation/visualization scripts.

## Files
- `main.py`: Starter entry point.
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

# Export all notebooks to HTML (run this after notebook updates)
py -3 export_notebooks_html.py

# Export a specific notebook to HTML
py -3 export_notebooks_html.py chip_model_unified_flow.ipynb
```


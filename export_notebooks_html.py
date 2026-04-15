from __future__ import annotations

from pathlib import Path
import argparse
import re
from urllib.parse import unquote

import nbformat
from nbconvert import HTMLExporter


def normalize_fragment_ids(html: str) -> str:
    id_pattern = re.compile(r'id="([^"]+)"')
    href_pattern = re.compile(r'href="#([^"]+)"')

    ids = set(id_pattern.findall(html))
    mapping: dict[str, str] = {}
    for old in ids:
        new = unquote(old)
        if new != old:
            mapping[old] = new

    for old, new in mapping.items():
        html = html.replace(f'id="{old}"', f'id="{new}"')
        html = html.replace(f'href="#{old}"', f'href="#{new}"')

    def _decode_href(m: re.Match[str]) -> str:
        frag = m.group(1)
        return f'href="#{unquote(frag)}"'

    return href_pattern.sub(_decode_href, html)


def fix_common_mojibake(html: str) -> str:
    replacements = {
        "Ã—": "×",
        "Â¶": "¶",
        "Â—": "—",
        "Â–": "–",
        "Â“": "“",
        "Â”": "”",
        "Â’": "’",
    }
    for bad, good in replacements.items():
        html = html.replace(bad, good)
    return html


def strip_legacy_contents_nav(html: str) -> str:
    html = re.sub(
        r"<style>\s*#notebook-contents\s*\{.*?\.jp-RenderedImage img \{.*?</style>",
        "",
        html,
        flags=re.DOTALL,
    )
    html = re.sub(
        r'<nav id="notebook-contents" class="notebook-contents">.*?</nav>',
        "",
        html,
        flags=re.DOTALL,
    )
    return html


def remove_anchor_links(html: str) -> str:
    # Drop Jupyter heading anchor glyph links entirely to prevent mojibake artifacts.
    return re.sub(r'<a class="anchor-link" href="#[^"]*">.*?</a>', "", html)


def export_one(ipynb_path: Path) -> Path:
    with ipynb_path.open("r", encoding="utf-8-sig") as f:
        nb = nbformat.read(f, as_version=4)

    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(nb)

    body = fix_common_mojibake(body)
    body = normalize_fragment_ids(body)
    body = strip_legacy_contents_nav(body)
    body = remove_anchor_links(body)

    out_html = ipynb_path.with_suffix(".html")
    out_html.write_text(body, encoding="utf-8")
    return out_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one or more notebooks to HTML.")
    parser.add_argument("notebooks", nargs="*", help="Notebook paths. If omitted, exports all *.ipynb in cwd.")
    args = parser.parse_args()

    if args.notebooks:
        targets = [Path(p) for p in args.notebooks]
    else:
        targets = sorted(Path(".").glob("*.ipynb"))

    if not targets:
        print("No notebooks found.")
        return

    for nb_path in targets:
        if not nb_path.exists():
            print(f"Skipping missing notebook: {nb_path}")
            continue
        out = export_one(nb_path)
        print(f"Exported {nb_path} -> {out}")


if __name__ == "__main__":
    main()

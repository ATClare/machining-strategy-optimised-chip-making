from __future__ import annotations

from pathlib import Path
import argparse
import re
from urllib.parse import unquote
from html import unescape

import nbformat
from nbconvert import HTMLExporter


def normalize_fragment_ids(html: str) -> str:
    id_pattern = re.compile(r'id=\"([^\"]+)\"')
    href_pattern = re.compile(r'href=\"#([^\"]+)\"')

    ids = set(id_pattern.findall(html))
    mapping: dict[str, str] = {}
    for old in ids:
        new = unquote(old)
        if new != old:
            mapping[old] = new

    for old, new in mapping.items():
        html = html.replace(f'id=\"{old}\"', f'id=\"{new}\"')
        html = html.replace(f'href=\"#{old}\"', f'href=\"#{new}\"')

    # Also decode any remaining internal href fragments for consistency.
    def _decode_href(m: re.Match[str]) -> str:
        frag = m.group(1)
        return f'href=\"#{unquote(frag)}\"'

    html = href_pattern.sub(_decode_href, html)
    html = html.replace("Â¶", "¶")
    return html


def fix_common_mojibake(html: str) -> str:
    # Common UTF-8/Windows-1252 mojibake seen in exported notebook text.
    replacements = {
        "\u00c3\u2014": "\u00d7",   # Ã— -> ×
        "Ã—": "×",
        "Â ": " ",
        "Â¶": "¶",
    }
    for bad, good in replacements.items():
        html = html.replace(bad, good)
    return html


def inject_contents_nav(html: str) -> str:
    heading_pattern = re.compile(
        r"<h([2-3]) id=\"([^\"]+)\">(.*?)</h\1>",
        re.DOTALL,
    )
    headings: list[tuple[int, str, str]] = []
    for level, frag_id, inner in heading_pattern.findall(html):
        text = re.sub(r"<[^>]+>", "", inner).strip()
        text = unescape(text)
        text = text.replace("¶", "").strip()
        if not text:
            continue
        headings.append((int(level), frag_id, text))

    if not headings:
        return html

    items: list[str] = []
    for level, frag_id, text in headings:
        cls = "toc-h2" if level == 2 else "toc-h3"
        items.append(f'<li class="{cls}"><a href="#{frag_id}">{text}</a></li>')
    nav = (
        '<nav id="notebook-contents" class="notebook-contents">'
        "<h2>Contents</h2>"
        '<ul class="notebook-contents-list">'
        + "".join(items)
        + "</ul></nav>"
    )

    style = """
<style>
#notebook-contents { border: 1px solid #d0d7de; border-radius: 8px; padding: 12px 14px; margin: 12px 0 18px; background: #f8fafc; }
#notebook-contents h2 { margin: 0 0 8px; font-size: 1.15rem; }
.notebook-contents-list { margin: 0; padding-left: 1.15rem; }
.notebook-contents-list li { margin: 3px 0; }
.notebook-contents-list li.toc-h3 { margin-left: 0.8rem; }
.jp-RenderedImage img { display: block; max-width: 100%; height: auto; }
</style>
"""

    # Insert contents block before the first H2.
    first_h2_match = re.search(r"<h2 id=\"", html)
    if first_h2_match:
        insert_at = first_h2_match.start()
        return html[:insert_at] + style + nav + html[insert_at:]

    # Fallback: append at the beginning of body.
    body_match = re.search(r"<body[^>]*>", html)
    if body_match:
        insert_at = body_match.end()
        return html[:insert_at] + style + nav + html[insert_at:]
    return style + nav + html


def export_one(ipynb_path: Path) -> Path:
    with ipynb_path.open("r", encoding="utf-8-sig") as f:
        nb = nbformat.read(f, as_version=4)

    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(nb)

    body = fix_common_mojibake(body)
    body = normalize_fragment_ids(body)
    body = inject_contents_nav(body)

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

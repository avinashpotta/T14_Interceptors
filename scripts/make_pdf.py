#!/usr/bin/env python3
"""Render docs/REPORT.md to a print-ready PDF.

    python scripts/make_pdf.py

Markdown -> HTML (markdown-it-py, tables enabled) -> PDF (WeasyPrint). No
LaTeX, no pandoc, no headless browser - all three are absent on this machine
and none is worth a multi-gigabyte install for one document.
"""
from __future__ import annotations

import sys
from pathlib import Path

from markdown_it import MarkdownIt
from weasyprint import HTML, CSS

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "REPORT.md"
DST = ROOT / "docs" / "AIPP_Team14_Interceptors_Report.pdf"

CSS_TEXT = """
@page {
  size: A4;
  margin: 20mm 18mm 18mm 18mm;
  @bottom-center {
    content: counter(page);
    font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
    font-size: 8.5pt; color: #8a8984;
  }
}
@page :first { @bottom-center { content: ""; } }

body {
  font-family: "Georgia", "Palatino", serif;
  font-size: 10pt; line-height: 1.48; color: #15150f;
  hyphens: auto;
}
h1 {
  font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  font-size: 20pt; line-height: 1.2; margin: 0 0 2mm 0; color: #0b0b0b;
}
h2 {
  font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  font-size: 13.5pt; margin: 9mm 0 2.5mm 0; color: #0b0b0b;
  padding-bottom: 1.2mm; border-bottom: 0.4pt solid #d8d7d2;
  break-after: avoid;
}
h3 {
  font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  font-size: 11pt; margin: 6mm 0 1.8mm 0; color: #0b0b0b;
  break-after: avoid;
}
h4 {
  font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  font-size: 10pt; margin: 4.5mm 0 1.5mm 0; color: #2f2e2a; break-after: avoid;
}
p { margin: 0 0 2.4mm 0; }
strong { color: #000; }

/* tables: compact, ruled only where it helps the eye */
table {
  border-collapse: collapse; width: 100%;
  margin: 2.5mm 0 3.5mm 0; font-size: 8.6pt;
  font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  break-inside: avoid;
}
th {
  text-align: left; font-weight: 600; color: #0b0b0b;
  border-bottom: 0.8pt solid #4a4945; padding: 1.4mm 2mm 1.2mm 0;
}
td { padding: 1.1mm 2mm 1.1mm 0; border-bottom: 0.3pt solid #e5e4e0; }
td:first-child, th:first-child { padding-left: 0; }

/* code */
code {
  font-family: "SF Mono", "Menlo", monospace; font-size: 8.4pt;
  background: #f3f2ee; padding: 0.3mm 0.8mm; border-radius: 1.2pt;
}
pre {
  background: #f7f6f2; border-left: 2pt solid #d8d7d2;
  padding: 2.2mm 3mm; margin: 2.5mm 0 3.5mm 0;
  font-size: 8.2pt; line-height: 1.38; break-inside: avoid;
  white-space: pre-wrap;
}
pre code { background: none; padding: 0; font-size: inherit; }

blockquote {
  margin: 3mm 0; padding: 0 0 0 4mm; border-left: 2pt solid #c7d7ee;
  color: #45443f; font-style: italic;
}
img { max-width: 100%; display: block; margin: 3mm auto 4mm auto; }
ul, ol { margin: 0 0 2.6mm 0; padding-left: 5mm; }
li { margin-bottom: 0.9mm; }
hr { border: none; border-top: 0.4pt solid #d8d7d2; margin: 6mm 0; }
a { color: #15150f; text-decoration: none; }

/* keep a heading with the figure or table that follows it */
h2 + p, h3 + p { break-before: avoid; }
"""


def main() -> None:
    if not SRC.exists():
        sys.exit(f"missing {SRC}")
    md = MarkdownIt("commonmark", {"html": True})\
        .enable("table").enable("strikethrough")
    html_body = md.render(SRC.read_text())
    html = f"<!doctype html><meta charset='utf-8'><body>{html_body}</body>"

    HTML(string=html, base_url=str(SRC.parent)).write_pdf(
        DST, stylesheets=[CSS(string=CSS_TEXT)])
    kb = DST.stat().st_size / 1024
    print(f"  wrote {DST.relative_to(ROOT)}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()

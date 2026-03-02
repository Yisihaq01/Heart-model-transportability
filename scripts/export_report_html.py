#!/usr/bin/env python3
"""
Export evaluation report from Markdown to HTML.
Reads reports/evaluation_report.md and writes reports/evaluation_report.html.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "reports" / "evaluation_report.md"
HTML_PATH = ROOT / "reports" / "evaluation_report.html"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Evaluation Report — Heart Disease Model Transportability</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; line-height: 1.6; }}
    h1 {{ border-bottom: 1px solid #ccc; padding-bottom: 0.3em; }}
    h2 {{ margin-top: 1.5em; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
    th, td {{ border: 1px solid #ddd; padding: 0.5em 0.75em; text-align: left; }}
    th {{ background: #f5f5f5; font-weight: 600; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    code {{ background: #f0f0f0; padding: 0.15em 0.4em; border-radius: 3px; font-size: 0.9em; }}
    pre {{ background: #f5f5f5; padding: 1em; overflow-x: auto; border-radius: 4px; }}
    a {{ color: #0066cc; }}
    .probast-link {{ margin-top: 1em; padding: 0.75em; background: #f8f9fa; border-left: 4px solid #0066cc; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> int:
    if not MD_PATH.exists():
        print(f"Error: {MD_PATH} not found. Run evaluation first.", file=sys.stderr)
        return 1

    try:
        import markdown
    except ImportError:
        print("Install markdown: pip install markdown", file=sys.stderr)
        return 1

    md_text = MD_PATH.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code"],
        output_format="html5",
    )
    html_doc = HTML_TEMPLATE.format(body=html_body)
    HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    HTML_PATH.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {HTML_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

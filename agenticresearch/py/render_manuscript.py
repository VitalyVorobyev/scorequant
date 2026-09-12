# /// script
# requires-python = ">=3.11"
# dependencies = ["markdown>=3.7"]
# ///
# ruff: noqa: E501  (the embedded stylesheet is lifted verbatim from v8.html)
r"""Render a manuscript Markdown source to its HTML sibling.

The manuscripts under ``agenticresearch/manuscripts/`` are authored in Markdown
(MathJax delimiters ``\(...\)`` and ``\[...\]``, result boxes as
``<div class="theorem" markdown="1">``, numeric citations ``[n]`` resolved
against ``<span id="ref-n">`` anchors in the References list, novelty tags
``[novelty: <label>; ledger <row>]`` rendered as superscript provenance marks that stay
hidden until the sidebar's "Show provenance" button is pressed). This script is the only sanctioned way
to produce the ``.html`` sibling, so the two files never drift: the stylesheet
below is the v8 stylesheet lifted verbatim, MathJax 3 is loaded from its CDN,
and the sidebar table of contents is generated from the ``##`` headings.

Usage::

    uv run agenticresearch/py/render_manuscript.py agenticresearch/manuscripts/<name>.md
    uv run agenticresearch/py/render_manuscript.py <name>.md -o <out>.html

The ``markdown`` dependency is declared inline (PEP 723) so ``uv run`` fetches
it on demand; it is deliberately not part of the library environment. Figures
must be referenced by relative path (``figures/...``); ``registry.py validate``
rejects inline ``data:`` images.
"""

from __future__ import annotations

import argparse
import html as html_lib
import re
import sys
from pathlib import Path

import markdown
from markdown.extensions.toc import slugify as _slugify_base

_STYLE = """:root{
  --bg:#f7f7f5; --paper:#ffffff; --ink:#171717; --muted:#646464; --line:#deded8;
  --accent:#263f73; --accent2:#6a3a19; --soft:#f0f2f7; --soft2:#f7f1ec;
  --green:#295f45; --red:#8a3434;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.62}
a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 290px;max-width:1320px;margin:0 auto;gap:34px;padding:34px}
main{background:var(--paper);border:1px solid var(--line);border-radius:18px;padding:58px 68px 72px;box-shadow:0 8px 28px rgba(0,0,0,.035)}
aside{position:sticky;top:22px;align-self:start;max-height:calc(100vh - 44px);overflow:auto;padding:18px 16px;border-left:1px solid var(--line)}
aside .toc-title{font-size:.78rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:700;margin-bottom:10px}
aside a{display:block;color:#4d4d4d;font-size:.88rem;padding:4px 6px;border-radius:6px}
aside a:hover{background:#ecece8;text-decoration:none;color:#111}
h1{font-family:Georgia,"Times New Roman",serif;font-size:3.05rem;line-height:1.08;margin:0 0 12px;letter-spacing:-.025em;font-weight:600}
.subtitle{font-family:Georgia,"Times New Roman",serif;font-size:1.34rem;color:#484848;margin-bottom:28px}
.meta{display:flex;flex-wrap:wrap;gap:8px 12px;margin-bottom:34px;font-size:.84rem;color:var(--muted)}
.tag{display:inline-flex;align-items:center;border:1px solid var(--line);border-radius:999px;padding:4px 10px;background:#fafafa}
.abstract{border-top:2px solid #222;border-bottom:1px solid var(--line);padding:25px 0;margin:24px 0 36px}
.abstract h2{font-size:.85rem;text-transform:uppercase;letter-spacing:.11em;margin:0 0 9px;font-family:inherit}
.abstract p{font-family:Georgia,"Times New Roman",serif;font-size:1.08rem;margin:0}
.abstract p+p{margin-top:12px}
mjx-container[jax="SVG"][display="true"]{min-width:0!important;max-width:100%;overflow-x:auto;overflow-y:hidden}
h2{font-family:Georgia,"Times New Roman",serif;font-size:2.0rem;margin:58px 0 15px;line-height:1.18;font-weight:600;border-top:1px solid var(--line);padding-top:28px}
h3{font-family:Georgia,"Times New Roman",serif;font-size:1.38rem;margin:34px 0 10px;line-height:1.25}
h4{font-size:1rem;margin:24px 0 7px}
p{margin:12px 0}
.lead{font-family:Georgia,"Times New Roman",serif;font-size:1.17rem;line-height:1.65}
.small{font-size:.88rem;color:var(--muted)}
.note,.remark,.theorem,.proposition,.lemma,.warning,.result{border-radius:11px;padding:17px 19px;margin:20px 0;border:1px solid var(--line)}
.theorem{background:var(--soft);border-left:4px solid var(--accent)}
.proposition,.lemma{background:#fafafa;border-left:4px solid #555}
.note,.remark{background:#fbfaf5;border-left:4px solid #987d3f}
.warning{background:#fff6f2;border-left:4px solid var(--accent2)}
.result{background:#f3f8f4;border-left:4px solid var(--green)}
.box-title{font-weight:750;margin-bottom:7px}
.status-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}
.status-card{border:1px solid var(--line);border-radius:10px;padding:13px;background:#fafafa}
.status-card b{display:block;font-size:.86rem;margin-bottom:4px}
.status-card span{font-size:.81rem;color:var(--muted)}
table{border-collapse:collapse;width:100%;margin:18px 0 24px;font-size:.91rem}
th,td{border-bottom:1px solid var(--line);padding:9px 10px;text-align:left;vertical-align:top}
th{font-size:.8rem;text-transform:uppercase;letter-spacing:.04em;color:#555;background:#fafafa}
tr:last-child td{border-bottom:0}
.eq-label{float:right;color:var(--muted);font-size:.8rem}
figure{margin:28px auto 34px}
figure img{display:block;max-width:100%;height:auto;border:1px solid var(--line);border-radius:10px;background:#fff}
figcaption{font-size:.84rem;color:var(--muted);margin-top:8px;line-height:1.45}
.figure-pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.kicker{font-size:.78rem;text-transform:uppercase;letter-spacing:.1em;font-weight:750;color:var(--accent);margin-bottom:8px}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:#171717;color:#f2f2f2;border-radius:10px;padding:16px;overflow:auto;font-size:.83rem;line-height:1.5}
details{border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:15px 0;background:#fcfcfb}
summary{cursor:pointer;font-weight:700}
.refs li{margin:9px 0;padding-left:4px}
.callout-number{font-family:Georgia,serif;font-size:2.15rem;font-weight:700;color:var(--accent);line-height:1}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:22px}
hr{border:0;border-top:1px solid var(--line);margin:34px 0}
sup.ref a{font-size:.75em}
.downloads{display:flex;gap:10px;flex-wrap:wrap;margin:20px 0}
.button{display:inline-block;padding:8px 12px;border:1px solid #bdbdb7;border-radius:8px;background:#fff;font-size:.88rem;color:#222}
.button:hover{text-decoration:none;background:#f1f1ed}
@media(max-width:1000px){.layout{display:block;padding:12px}aside{display:none}main{padding:38px 30px;border-radius:12px}.status-grid{grid-template-columns:1fr 1fr}}
@media(max-width:1000px){table{display:block;max-width:100%;overflow-x:auto}mjx-container{max-width:100%;overflow-x:auto;overflow-y:hidden}mjx-container:not([display]){display:inline-block;vertical-align:middle}code{overflow-wrap:anywhere}}
@media(max-width:650px){main{padding:30px 20px}h1{font-size:2.2rem}.figure-pair,.two-col,.status-grid{grid-template-columns:1fr}.subtitle{font-size:1.12rem}table{font-size:.82rem}}
@media print{body{background:#fff}.layout{display:block;padding:0}aside{display:none}main{border:0;box-shadow:none;padding:0;max-width:none}.button{display:none}details{break-inside:avoid}details>summary{display:none}details>*{display:block!important}a{color:#000}}

.mode-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:22px 0}
.mode-card{border:1px solid var(--line);border-radius:12px;padding:16px;background:#fafafa}
.mode-card h4{font-family:Georgia,"Times New Roman",serif;font-size:1.08rem;margin:0 0 6px}
.mode-card p{font-size:.89rem;margin:0;color:#444}
.diagram{border:1px solid var(--line);border-radius:12px;background:#fcfcfb;padding:18px;margin:26px 0;overflow:auto}
.eqbox{background:#f8fafc;border:1px solid #d9dfe8;border-radius:10px;padding:15px 18px;margin:18px 0}
.badge{display:inline-block;font-size:.73rem;letter-spacing:.04em;text-transform:uppercase;border:1px solid var(--line);border-radius:999px;padding:2px 8px;margin-right:6px;color:#555;background:#fff}
.api-table td:first-child{white-space:nowrap;font-weight:650}
@media(max-width:850px){.mode-grid{grid-template-columns:1fr}}
sup.novelty{display:none;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.66rem;color:var(--muted);white-space:nowrap;margin-left:2px;cursor:help}
body.show-provenance sup.novelty{display:inline}
.provenance-toggle{display:block;width:100%;margin:18px 0 0;padding:7px 10px;font:inherit;font-size:.82rem;color:#444;background:#fafafa;border:1px solid var(--line);border-radius:8px;cursor:pointer}
.provenance-toggle:hover{background:#f0f0ee}
.placement td:first-child{white-space:nowrap;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.82rem}
"""

_MATHJAX = (
    "<script>\n"
    "window.MathJax = {tex: {inlineMath: [['\\\\(','\\\\)']], displayMath: [['\\\\[','\\\\]']]}, "
    "svg: {fontCache: 'global'}};\n"
    "</script>\n"
    '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>'
)

_PROVENANCE_TOGGLE = (
    '<button type="button" class="provenance-toggle" '
    "onclick=\"document.body.classList.toggle('show-provenance');"
    "this.textContent=document.body.classList.contains('show-provenance')"
    "?'Hide provenance':'Show provenance'\">Show provenance</button>"
)

_MATH = re.compile(r"\\\[.*?\\\]|\\\(.*?\\\)", re.DOTALL)
_CITE = re.compile(r"(?<![\w\\])\[(\d{1,3})\]")
_NOVELTY = re.compile(r"\s*\[novelty: ([^;\]]+); ledger ([^\]]+)\]")
_TITLE = re.compile(r"^# (.+)$", re.MULTILINE)
_DESCRIPTION = re.compile(r"^<!-- description: (.+?) -->$", re.MULTILINE)


def _protect_math(text: str) -> tuple[str, list[str]]:
    """Swap every math span for an inert token so Markdown never touches it."""
    spans: list[str] = []

    def keep(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return f"⟦MATH{len(spans) - 1}⟧"

    return _MATH.sub(keep, text), spans


def _restore_math(rendered: str, spans: list[str]) -> str:
    def back(match: re.Match[str]) -> str:
        return html_lib.escape(spans[int(match.group(1))], quote=False)

    return re.sub(r"⟦MATH(\d+)⟧", back, rendered)


def _link_citations(rendered: str) -> str:
    return _CITE.sub(lambda m: f'<a href="#ref-{m.group(1)}">[{m.group(1)}]</a>', rendered)


def _mark_novelty(rendered: str) -> str:
    """Render a novelty tag as a superscript provenance mark, hidden until toggled on.

    The whitespace before the tag is consumed so that hiding the mark leaves no stray
    space before the punctuation that follows it.
    """

    def mark(m: re.Match[str]) -> str:
        label, rows = m.group(1).strip(), m.group(2).strip()
        title = html_lib.escape(f"novelty: {label}; ledger {rows}", quote=True)
        return f'<sup class="novelty" title="{title}">{html_lib.escape(rows)}</sup>'

    return _NOVELTY.sub(mark, rendered)


def _wrap_abstract(rendered: str) -> str:
    """Give the Abstract heading the ``section.abstract`` wrapper the stylesheet expects."""
    start = rendered.find('<h2 id="abstract">')
    if start < 0:
        return rendered
    cut = len(rendered)
    for marker in ("<h2 ", '<div class="status-grid">'):
        pos = rendered.find(marker, start + 1)
        if pos >= 0:
            cut = min(cut, pos)
    return (
        rendered[:start]
        + '<section class="abstract">\n'
        + rendered[start:cut]
        + "</section>\n"
        + rendered[cut:]
    )


def _toc(md: markdown.Markdown) -> str:
    items = ['<div class="toc-title">Contents</div>']
    in_appendices = False
    for token in md.toc_tokens:
        if token["level"] != 2:
            continue
        name = html_lib.unescape(re.sub(r"<[^>]+>", "", token["name"]))
        name = re.sub(r"\\\((.*?)\\\)", r"\1", name)
        if name.startswith("Appendix") and not in_appendices:
            in_appendices = True
            items.append('<div class="toc-title">Appendices</div>')
        items.append(f'<a href="#{token["id"]}">{html_lib.escape(name)}</a>')
    items.append(_PROVENANCE_TOGGLE)
    return "\n".join(items)


def _slugify(value: str, separator: str) -> str:
    """Slugify a heading with its math placeholders removed."""
    return _slugify_base(re.sub(r"⟦MATH\d+⟧", "", value), separator)


def render(source: str) -> str:
    """Return the full HTML page for one manuscript Markdown source."""
    title_match = _TITLE.search(source)
    title = title_match.group(1).strip() if title_match else "Manuscript"
    description_match = _DESCRIPTION.search(source)
    description = description_match.group(1).strip() if description_match else ""
    protected, spans = _protect_math(source)
    md = markdown.Markdown(
        extensions=["extra", "toc", "sane_lists"],
        extension_configs={"toc": {"toc_depth": "2-3", "slugify": _slugify}},
        output_format="html",
    )
    body = md.convert(protected)
    body = _link_citations(body)
    body = _mark_novelty(body)
    body = _restore_math(body, spans)
    body = _wrap_abstract(body)
    toc = _restore_math(_toc(md), spans)
    plain_title = re.sub(r"\\\((.*?)\\\)", r"\1", title)
    head = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{html_lib.escape(plain_title)}</title>",
    ]
    if description:
        head.append(
            f'<meta name="description" content="{html_lib.escape(description, quote=True)}">'
        )
    head.extend([_MATHJAX, "<style>", _STYLE, "</style>", "</head>", "<body>"])
    return (
        "\n".join(head)
        + '\n<div class="layout">\n<main>\n'
        + body
        + "\n</main>\n<aside>\n"
        + toc
        + "\n</aside>\n</div>\n</body></html>\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Render the source named on the command line; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", type=Path, help="manuscript .md file")
    parser.add_argument("-o", "--output", type=Path, help="output .html (default: sibling)")
    args = parser.parse_args(argv)
    output = args.output or args.source.with_suffix(".html")
    text = args.source.read_text(encoding="utf-8")
    if "data:image/" in text:
        print("refusing to render: inline data: image found", file=sys.stderr)
        return 1
    output.write_text(render(text), encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

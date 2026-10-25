#!/usr/bin/env python3
"""Erzeugt snn-fpga/PUBLICATION.pdf aus snn-fpga/PUBLICATION.md.

    pip install markdown matplotlib
    python3 publication_pdf.py [PUBLICATION.md] [PUBLICATION.pdf]

Ablauf: Markdown -> HTML (Tabellen), LaTeX-Bloecke -> SVG (matplotlib mathtext),
ASCII-Systemskizze -> SVG-Abbildung, HTML -> PDF mit Chromium (Playwright, ueber node).
"""
import io
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown
import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "PUBLICATION.md"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else SRC.with_suffix(".pdf")

plt.rcParams.update({"mathtext.fontset": "stix", "svg.fonttype": "path"})


def latex_svg(tex):
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=13)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.03, transparent=True)
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.index("<svg"):]


def figure_svg():
    ink, quiet, edge, accent, tint = "#1b1b1b", "#555", "#6b6b6b", "#2f6fbd", "#f2f4f7"

    def box(x, y, w, h, main=False):
        if main:
            return (f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill='{accent}' "
                    f"fill-opacity='0.12' stroke='{accent}' stroke-width='2'/>")
        return (f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill='none' "
                f"stroke='{edge}' stroke-width='1.25'/>")

    def t(x, y, s, size=13, weight=400, fill=ink, anchor="start"):
        return (f"<text x='{x}' y='{y}' font-size='{size}' font-weight='{weight}' fill='{fill}' "
                f"text-anchor='{anchor}'>{s}</text>")

    def named(x, y, w, h, name, lines, main=False):
        out = [box(x, y, w, h, main), t(x + 12, y + 24, name, weight=600)]
        for k, line in enumerate(lines):
            out.append(t(x + 12, y + 44 + 16 * k, line, size=11.5, fill=ink if main else quiet))
        return "".join(out)

    arrows = ["M224 84H280", "M500 84H556", "M294 252V280", "M340 336H372", "M420 280V224H388"]
    both = ["M294 116V196", "M520 280V232H560", "M640 268V300"]
    paths = "".join(f"<path d='{d}' marker-end='url(#a)'/>" for d in arrows)
    paths += "".join(f"<path d='{d}' marker-start='url(#a)' marker-end='url(#a)'/>" for d in both)
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 760 420' font-family='Inter, sans-serif'>",
        f"<defs><marker id='a' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='6' markerHeight='6' "
        f"orient='auto-start-reverse'><path d='M0 0L10 5L0 10z' fill='{edge}'/></marker></defs>",
        f"<rect x='24' y='152' width='712' height='252' rx='8' fill='{tint}' stroke='{edge}' stroke-width='1.25'/>",
        t(40, 176, "Tang Nano 20K · GW2AR-18 · 54 MHz", weight=600),
        f"<g fill='none' stroke='{edge}' stroke-width='1.25'>{paths}</g>",
        named(24, 52, 200, 64, "Gas sensor", ["Wio Terminal, 4 channels"]),
        named(280, 52, 220, 64, "Host (PC or Lite3)", ["Features, calibration, ROS 2"]),
        named(556, 52, 180, 64, "Lite3 behaviour", ["Cast-and-surge"]),
        named(200, 196, 188, 56, "UART commands", ["W · R · B · L · C · Z"]),
        named(40, 280, 300, 112, "SNN core (gather)",
              ["Input spikes: xorshift32", "800 neurons, 19.5 cycles each on average",
               "LIF as in Norse 1.1.0, states Q8.7", "BSRAM: win0/1, syn0/1, pp, v, i, counts"], main=True),
        named(372, 280, 164, 80, "Readout", ["4 MAC per cycle", "Streaming argmax"]),
        named(560, 196, 160, 72, "SDRAM controller", ["CL 2, burst reads", "Calibrated capture"]),
        named(560, 300, 160, 72, "SDRAM 64 Mbit", ["W_out, 541 KiB", "in package"]),
        t(252, 76, "CSV", 11.5, fill=quiet, anchor="middle"),
        t(528, 76, "ROS 2", 11.5, fill=quiet, anchor="middle"),
        t(302, 140, "UART, 1 Mbaud", 11.5, fill=quiet),
        t(302, 272, "Rates, seed", 11.5, fill=quiet),
        t(426, 266, "Class, logits", 11.5, fill=quiet),
        t(540, 224, "Bursts", 11.5, fill=quiet, anchor="middle"),
        "</svg>",
    ]
    return "".join(parts)


CSS = """
@page { size: A4; margin: 22mm 20mm 20mm 20mm; }
body { font-family: 'FreeSerif', 'DejaVu Serif', serif; font-size: 10.6pt; line-height: 1.42;
       color: #111; text-align: justify; hyphens: auto; }
h1 { font-family: 'Inter', sans-serif; font-size: 19pt; line-height: 1.22; text-align: left;
     margin: 0 0 4pt 0; font-weight: 700; }
h1 + p { text-align: left; color: #555; margin-top: 0; }
h2 { font-family: 'Inter', sans-serif; font-size: 12.5pt; margin: 16pt 0 5pt; text-align: left;
     page-break-after: avoid; font-weight: 650; }
h2#abstract { margin-top: 10pt; }
p { margin: 0 0 6pt 0; orphans: 3; widows: 3; }
ul, ol { margin: 0 0 6pt 0; padding-left: 18pt; }
li { margin-bottom: 2pt; }
code { font-family: 'JetBrains Mono', monospace; font-size: 8.6pt; background: #f2f2f2;
       padding: 0 2pt; border-radius: 2pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 9pt; font-size: 9.2pt;
        page-break-inside: avoid; text-align: left; }
th { font-family: 'Inter', sans-serif; font-weight: 600; font-size: 8.6pt; border-bottom: 1.2pt solid #333;
     padding: 3pt 5pt; }
td { border-bottom: 0.5pt solid #ccc; padding: 3pt 5pt; vertical-align: top; }
.eq { text-align: center; margin: 7pt 0 9pt; page-break-inside: avoid; }
.eq svg { max-width: 100%; height: auto; }
figure { margin: 8pt 0 10pt; page-break-inside: avoid; }
figure svg { width: 100%; height: auto; }
figcaption { font-size: 9.2pt; color: #333; text-align: justify; margin-top: 3pt; }
a { color: #1d4f91; text-decoration: none; }
.abstract { background: #f6f7f9; padding: 8pt 11pt 3pt; border-left: 2.5pt solid #2f6fbd; }
"""


def build_html(md_text):
    eqs = []

    def take_eq(m):
        eqs.append(latex_svg(m.group(1).strip()))
        return f"\n\nEQPLACEHOLDER{len(eqs) - 1}\n\n"

    md_text = re.sub(r"```latex\n(.*?)\n```", take_eq, md_text, flags=re.S)
    fig_cap = ""
    m = re.search(r"```\n(Gas sensor.*?)\n```\n\n\*(Figure 1:.*?)\*", md_text, flags=re.S)
    if m:
        fig_cap = m.group(2)
        md_text = md_text.replace(m.group(0), "\n\nFIGPLACEHOLDER\n\n")
    html = markdown.markdown(md_text, extensions=["tables", "toc"])
    for i, svg in enumerate(eqs):
        html = html.replace(f"<p>EQPLACEHOLDER{i}</p>", f"<div class='eq'>{svg}</div>")
    html = html.replace("<p>FIGPLACEHOLDER</p>",
                        f"<figure>{figure_svg()}<figcaption>{markdown.markdown(fig_cap)[3:-4]}</figcaption></figure>")
    # Abstract hervorheben: von h2#abstract bis zum naechsten h2
    html = re.sub(r"(<h2 id=\"abstract\">Abstract</h2>)(.*?)(?=<h2)",
                  lambda mm: f"{mm.group(1)}<div class='abstract'>{mm.group(2)}</div>", html, flags=re.S)
    # Bezeichner wie W_in, tau_mem tiefstellen (nicht in Code, Grafiken und Link-Zielen)
    parts = re.split(r"(<code>.*?</code>|<svg.*?</svg>|<a [^>]*>)", html, flags=re.S)
    sub = re.compile(r"(?<![\w/])([A-Za-z\u03c4])_(in|eff|out|mem|syn|rec)\b")
    html = "".join(x if i % 2 else sub.sub(r"\1<sub>\2</sub>", x) for i, x in enumerate(parts))
    title = re.search(r"<h1[^>]*>(.*?)</h1>", html).group(1)
    return f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{title}</title>" \
           f"<style>{CSS}</style></head><body>{html}</body></html>", title


def main():
    html, title = build_html(SRC.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "paper.html"
        page.write_text(html, encoding="utf-8")
        js = f"""
const {{ chromium }} = require('playwright');
(async () => {{
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto('file://{page}', {{waitUntil: 'networkidle'}});
  await p.pdf({{ path: {json.dumps(str(OUT))}, format: 'A4', printBackground: true,
    displayHeaderFooter: true, headerTemplate: '<div></div>',
    footerTemplate: '<div style="width:100%;font-size:8px;color:#777;text-align:center;font-family:Inter,sans-serif"><span class=pageNumber></span> / <span class=totalPages></span></div>',
    margin: {{ top: '22mm', bottom: '20mm', left: '20mm', right: '20mm' }} }});
  await b.close();
}})();
"""
        (Path(tmp) / "print.js").write_text(js)
        subprocess.run(["node", str(Path(tmp) / "print.js")], check=True)
    print(f"{OUT} geschrieben ({title})")


if __name__ == "__main__":
    main()

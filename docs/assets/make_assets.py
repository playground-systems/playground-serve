"""Writes the README's pictures: model-time-light.svg and model-time-dark.svg (stock against fast path model time per
model family, every number read from MODELS.md) and quickstart.svg (a terminal with real `pgs` output).

    python3 docs/assets/make_assets.py

Plain SVG shapes and text with explicit colors: GitHub shows README images without scripts or outside CSS.
"""

from __future__ import annotations

import pathlib
import re
from html import escape

HERE = pathlib.Path(__file__).resolve().parent
MODELS_MD = HERE.parents[1] / "MODELS.md"

# --- chart: model time per family, from MODELS.md ------------------------------------------------------------------

# (MODELS.md table entry, label). One entry per family that has a fast path; FLUX 3 Action has none (stock only).
FAMILY_ROWS = [
    ("molmoact2_droid", "MolmoAct2"),
    ("xiaomi_r1_robocasa365", "Xiaomi-Robotics-1"),
    ("abc_vla_yam", "ABC-VLA"),
    ("flex_pi_libero", "Flex-π"),
    ("gr00t_n17_libero", "GR00T N1.7"),
    ("cosmos3_edge_droid", "Cosmos 3 Edge"),
]

THEMES = {
    "light": dict(surface="#ffffff", border="#d1d9e0", text="#1f2328", secondary="#59636e", muted="#818b98",
                  grid="#eaeef2", axis="#c8d1da", stock="#8c959f", fast="#2a78d6"),
    "dark": dict(surface="#0d1117", border="#3d444d", text="#f0f6fc", secondary="#9198a1", muted="#7d8590",
                 grid="#21262d", axis="#3d444d", stock="#656c76", fast="#3987e5"),
}
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"


def read_rows() -> list[dict]:
    """Each row: label, robot, GPU, (stock low, high), (fast low, high), from MODELS.md as it is now."""
    text = MODELS_MD.read_text()
    rows = []
    for entry, label in FAMILY_ROWS:
        line = next(ln for ln in text.splitlines() if ln.startswith(f"| `{entry}`"))
        cells = [c.strip() for c in line.strip("|").split("|")]
        stock, fast = map(float, re.match(r"([\d.]+) / ([\d.]+)", cells[6]).groups())
        rows.append(dict(label=label, robot=cells[1], gpu=cells[4].split(",")[0],
                         experimental="(experimental)" in cells[0], stock=(stock, stock), fast=(fast, fast)))
    # pi05_libero (openpi) is in the openpi section: "served `infer_ms` 31 to 34 ms against 75 to 79 for stock".
    m = re.search(r"served `infer_ms` (\d+) to (\d+) ms against (\d+) to (\d+) for stock", text)
    f_lo, f_hi, s_lo, s_hi = map(float, m.groups())
    assert "fast** (default for `pi05_libero` on an RTX 4090" in text
    rows.append(dict(label="π0.5 (openpi)", robot="LIBERO", gpu="RTX 4090", experimental=False,
                     stock=(s_lo, s_hi), fast=(f_lo, f_hi)))
    for r in rows:
        r["speedup"] = (r["stock"][0] / r["fast"][1], r["stock"][1] / r["fast"][0])
    return sorted(rows, key=lambda r: -sum(r["speedup"]))


def num(lo: float, hi: float) -> str:
    """One value as MODELS.md gives it (one decimal), or a range (whole milliseconds there)."""
    return f"{lo:.1f}" if lo == hi else f"{lo:.0f}–{hi:.0f}"


def bar(x0: float, x1: float, y: float, h: float, fill: str, opacity: float = 1.0) -> str:
    """A horizontal bar from the baseline x0: square at the baseline, 4 px rounded data end."""
    r = min(4.0, (x1 - x0) / 2, h / 2)
    op = f' fill-opacity="{opacity}"' if opacity < 1 else ""
    return (f'<path d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} A{r},{r} 0 0 1 {x1:.1f},{y + r:.1f} V{y + h - r:.1f} '
            f'A{r},{r} 0 0 1 {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z" fill="{fill}"{op}/>')


def chart(theme: str, rows: list[dict]) -> str:
    c = THEMES[theme]
    W, pad, row_h = 880, 28, 52
    x0, x_max, axis_max = 236, 760, 600.0
    sx = (x_max - x0) / axis_max
    top = 128
    H = top + row_h * len(rows) + 88
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'font-family="{SANS}" role="img" aria-labelledby="t d">',
           '<title id="t">Model time per call, stock against the Playground Serve fast path, per model family</title>',
           '<desc id="d">' + escape("; ".join(
               f"{r['label']} on {r['robot']}, {r['gpu']}: stock {num(*r['stock'])} ms, fast {num(*r['fast'])} ms"
               for r in rows)) + '</desc>',
           f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="12" fill="{c["surface"]}" '
           f'stroke="{c["border"]}"/>',
           f'<text x="{pad}" y="42" font-size="18" font-weight="600" fill="{c["text"]}">'
           'Time per model call: stock against the fast path</text>',
           f'<text x="{pad}" y="66" font-size="13" fill="{c["secondary"]}">Server infer_ms, median, on each '
           "model's default GPU. Same weights, precision, steps and inputs;</text>",
           f'<text x="{pad}" y="84" font-size="13" fill="{c["secondary"]}">the fast path must reproduce the '
           "stock actions at every start, or the worker serves stock.</text>"]
    # legend
    lx = pad
    for key, name in (("stock", "stock: the model's own inference code"), ("fast", "Playground Serve fast path")):
        out.append(f'<rect x="{lx}" y="103" width="12" height="12" rx="3" fill="{c[key]}"/>')
        out.append(f'<text x="{lx + 18}" y="113.5" font-size="12.5" fill="{c["secondary"]}">{escape(name)}</text>')
        lx += 18 + 7.0 * len(name) + 22
    out.append(f'<text x="{W - pad}" y="113.5" font-size="12.5" text-anchor="end" fill="{c["secondary"]}">'
               'faster</text>')
    # grid and axis
    plot_bottom = top + row_h * len(rows)
    for t in range(0, int(axis_max) + 1, 100):
        x = x0 + t * sx
        out.append(f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{plot_bottom}" '
                   f'stroke="{c["axis"] if t == 0 else c["grid"]}" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{plot_bottom + 18}" font-size="11.5" text-anchor="middle" '
                   f'fill="{c["muted"]}">{t}{" ms" if t == axis_max else ""}</text>')
    for i, r in enumerate(rows):
        y = top + i * row_h + 6
        sub = f"{r['robot']} · {r['gpu']}" + (" · experimental" if r["experimental"] else "")
        out.append(f'<text x="{pad}" y="{y + 13}" font-size="14" font-weight="600" fill="{c["text"]}">'
                   f'{escape(r["label"])}</text>')
        out.append(f'<text x="{pad}" y="{y + 31}" font-size="12" fill="{c["secondary"]}">{escape(sub)}</text>')
        for j, key in enumerate(("stock", "fast")):
            lo, hi = r[key]
            by = y + j * 16
            out.append(bar(x0, x0 + lo * sx, by, 13, c[key]))
            if hi > lo:  # a measured range: solid to the low end, lighter to the high end
                out.append(bar(x0 + lo * sx - 4, x0 + hi * sx, by, 13, c[key], 0.4))
            ink = c["text"] if key == "fast" else c["secondary"]
            out.append(f'<text x="{x0 + hi * sx + 6:.1f}" y="{by + 11}" font-size="12" fill="{ink}" '
                       f'font-weight="{600 if key == "fast" else 400}">{num(lo, hi)}</text>')
        lo, hi = r["speedup"]
        sp = f"{lo:.1f}x" if f"{lo:.1f}" == f"{hi:.1f}" else f"{lo:.1f}–{hi:.1f}x"
        out.append(f'<text x="{W - pad}" y="{y + 21}" font-size="16" font-weight="600" text-anchor="end" '
                   f'fill="{c["text"]}">{sp}</text>')
    out.append(f'<text x="{pad}" y="{H - 38}" font-size="11.5" fill="{c["muted"]}">Every number from MODELS.md, '
               "measured on Runpod Serverless from 2026-11-13 to 2026-11-16.</text>")
    out.append(f'<text x="{pad}" y="{H - 20}" font-size="11.5" fill="{c["muted"]}">Not shown: FLUX 3 Action '
               "(experimental), which serves its stock path only, about 3.2 s per call.</text>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --- terminal: real output of `pgs`, trimmed ----------------------------------------------------------------------

# Lines copied from the saved output of one real session, 2026-11-16: `pgs up pi05_libero` (a first start on a new
# volume, endpoint in US-NC-1), `pgs demo` and `pgs down`, client on the US east coast. Trimmed: lines left out, "…"
# marks a cut inside a line, and the owner and resource ids are shortened. Kinds: p = prompt, o = output, k =
# highlighted output.
TERMINAL = [
    ("p", "pgs up pi05_libero", ""),
    ("o", "GPU:    RTX 4090 (pool ADA_24) in US-NC-1, serverless stock LOW: 1.34 USD/hr while a worker runs; …", ""),
    ("o", "Data center: US-NC-1, 24.4 ms round trip from here (probe)", ""),
    ("o", "Volume: 40 GB STANDARD in US-NC-1: about 2.80 USD/month until deleted", ""),
    ("o", "Created volume playground-serve-o24a…-pi05-libero-002ef9 (ye3r…).", ""),
    ("o", "Created endpoint playground-serve-o24a…-pi05-libero-de5ce3 (tq6e…), no worker kept: …", ""),
    ("o", "  [0:04] starting the container (the first time on a host it pulls the image, about 2 minutes)", ""),
    ("o", "  [0:52] downloading the checkpoint", ""),
    ("o", "  [1:28] converting the checkpoint for the fast path", ""),
    ("o", "", ""),
    ("k", "Ready after 1:44.", ""),
    ("o", "Websocket: wss://tq6e….api.runpod.ai/ws", ""),
    ("o", "Billing: the GPU bills 1.34 USD/hr only while a worker runs. …", ""),
    ("o", "", ""),
    ("p", "pgs demo", ""),
    ("o", "Connecting to pi05_libero (if no worker runs, one starts now: about 1 minute)...", ""),
    ("k", "Actions: shape (10, 7) (horizon x action size; the observation was random, so the values mean nothing).", ""),
    ("o", "First action after 0.5 s (includes connecting and any worker start).", ""),
    ("o", "Round trip over 5 more calls: median 91 ms, max 147 ms; model time on the server median 34 ms.", ""),
    ("o", "", ""),
    ("p", "pgs down pi05_libero", ""),
    ("o", "  The worker is still saving the environment and checkpoint to the volume; waiting so that …", ""),
    ("o", "pi05_libero: no worker kept … once no client is connected, and GPU billing stops then.", ""),
    ("o", "  volume … kept: about 2.80 USD/month (`pgs delete pi05_libero --volume` removes it). …", ""),
]
COLUMNS = 106


def terminal() -> str:
    assert all(len(t) + (len(n) + 4 if n else 0) <= COLUMNS for _, t, n in TERMINAL), "line too long"
    fs, lh, cw = 13, 19, 7.85  # font size, line height, a generous monospace advance (0.6 em and some room)
    pad, bar_h = 20, 34
    W = round(2 * pad + COLUMNS * cw)
    foot = 2
    H = bar_h + 16 + lh * len(TERMINAL) + 14 + 18 * foot + 12
    bg, chrome, border = "#0d1117", "#161b22", "#30363d"
    ink = {"o": "#c9d1d9", "k": "#3fb950", "cmd": "#f0f6fc", "prompt": "#58a6ff", "note": "#8b949e"}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           f'role="img" aria-labelledby="t">',
           '<title id="t">Terminal: pgs up pi05_libero, pgs demo and pgs down, real output, trimmed</title>',
           f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="10" fill="{bg}" stroke="{border}"/>',
           f'<path d="M0.5,{bar_h} V10.5 A10,10 0 0 1 10.5,0.5 H{W - 10.5} A10,10 0 0 1 {W - 0.5},10.5 V{bar_h} Z" '
           f'fill="{chrome}"/>',
           f'<line x1="0.5" y1="{bar_h}" x2="{W - 0.5}" y2="{bar_h}" stroke="{border}"/>']
    for k, color in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        out.append(f'<circle cx="{pad + 2 + 20 * k}" cy="{bar_h / 2}" r="6" fill="{color}"/>')
    out.append(f'<text x="{W / 2}" y="{bar_h / 2 + 4.5}" font-family="{SANS}" font-size="12.5" text-anchor="middle" '
               f'fill="#8b949e">pgs, first start of pi05_libero on a new volume</text>')
    out.append(f'<g font-family="{MONO}" font-size="{fs}">')
    y = bar_h + 16 + 13

    def keep(s: str) -> str:  # no-break spaces: every renderer keeps them, with the same advance in monospace
        return escape(s).replace(" ", "\u00a0")

    for kind, text, note in TERMINAL:
        if kind == "p":
            line = (f'<tspan fill="{ink["prompt"]}">{keep("$ ")}</tspan><tspan fill="{ink["cmd"]}" '
                    f'font-weight="600">{keep(text)}</tspan>')
            if note:
                line += f'<tspan fill="{ink["note"]}">{keep("    " + note)}</tspan>'
        else:
            line = keep(text)
        color = ink["k"] if kind == "k" else ink["o"]
        out.append(f'<text x="{pad}" y="{y}" fill="{color}">{line}</text>' if text else "")
        y += lh
    out.append("</g>")
    y += 8
    out.append(f'<line x1="{pad}" y1="{y - 12}" x2="{W - pad}" y2="{y - 12}" stroke="{border}"/>')
    for text in ("Real output of one pgs session on 2026-11-16, trimmed: … marks a cut, ids are shortened.",
                 "A first start on a new volume in US-NC-1; the client on the US east coast."):
        out.append(f'<text x="{pad}" y="{y + 6}" font-family="{SANS}" font-size="12" fill="{ink["note"]}">'
                   f'{escape(text)}</text>')
        y += 18
    out.append("</svg>")
    return "\n".join(s for s in out if s) + "\n"


if __name__ == "__main__":
    rows = read_rows()
    for theme in THEMES:
        (HERE / f"model-time-{theme}.svg").write_text(chart(theme, rows))
    (HERE / "quickstart.svg").write_text(terminal())
    for r in rows:
        print(f"{r['label']:20} {r['robot']:16} {r['gpu']:9} stock {num(*r['stock']):>9}  fast {num(*r['fast']):>9}  "
              f"{r['speedup'][0]:.2f} to {r['speedup'][1]:.2f}x")

"""Writes social-preview.png (1280 x 640), the picture GitHub shows when a link to the repository is shared. The chart
takes its numbers from MODELS.md through make_assets.read_rows(). Pillow (a dependency of the package) and a system
font: Helvetica Neue and Menlo on macOS, DejaVu on Linux.

    uv run python docs/assets/make_social_preview.py

GitHub takes the picture only from the repository's settings (Settings, General, Social preview), uploaded by hand.
"""

from __future__ import annotations

import pathlib

from PIL import Image, ImageDraw, ImageFont

from make_assets import THEMES, read_rows

HERE = pathlib.Path(__file__).resolve().parent
W, H, SCALE = 1280, 640, 2  # drawn at twice the size, then shrunk, for smooth edges
C = THEMES["dark"]
FONTS = {  # (path, index in a .ttc) candidates per style, first found wins
    "bold": [("/System/Library/Fonts/HelveticaNeue.ttc", 1), ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0)],
    "regular": [("/System/Library/Fonts/HelveticaNeue.ttc", 0), ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0)],
    "mono": [("/System/Library/Fonts/Menlo.ttc", 0), ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 0)],
}


def font(style: str, size: float) -> ImageFont.FreeTypeFont:
    for path, index in FONTS[style]:
        if pathlib.Path(path).exists():
            return ImageFont.truetype(path, round(size * SCALE), index=index)
    raise SystemExit(f"no {style} font found; add one to FONTS")


def main() -> None:
    img = Image.new("RGB", (W * SCALE, H * SCALE), C["surface"])
    d = ImageDraw.Draw(img)

    def text(x, y, s, style, size, fill, anchor="la"):
        d.text((x * SCALE, y * SCALE), s, font=font(style, size), fill=fill, anchor=anchor)

    def rect(x0, y0, x1, y1, fill, r=0):
        d.rounded_rectangle([x0 * SCALE, y0 * SCALE, x1 * SCALE, y1 * SCALE], radius=r * SCALE, fill=fill)

    left = 72
    text(left, 52, "Playground Serve", "bold", 80, C["text"])
    text(left, 154, "Low-cost, open source inference for robot policies (VLAs)", "regular", 34, C["secondary"])

    # left column: the command and three facts from the README
    y = 240
    rect(left, y, left + 392, y + 58, "#161b22", r=10)
    text(left + 22, y + 29, "$ pgs up pi05_libero", "mono", 28, C["text"], anchor="lm")
    for i, line in enumerate(("Serverless GPU in your own account",
                              "Eight model families, 23 entries",
                              "Scales to zero when the robot stops",
                              "Apache 2.0")):
        text(left, 336 + i * 50, line, "regular", 27, C["text"] if i < 3 else C["muted"])

    # right column: model time per call, stock against the fast path
    rows = read_rows()
    cx, label_w, bar_w, right = 620, 206, 274, W - 72
    text(cx, 240, "Model time per call", "bold", 24, C["text"])
    lx = cx
    for key, name in (("stock", "stock"), ("fast", "fast path")):
        rect(lx, 280, lx + 16, 296, C[key], r=3)
        text(lx + 24, 288, name, "regular", 19, C["secondary"], anchor="lm")
        lx += 24 + 10.5 * len(name) + 26
    axis_max = max(r["stock"][1] for r in rows)
    y = 318
    for r in rows:
        text(cx, y + 14, r["label"], "regular", 21, C["text"], anchor="lm")
        x0 = cx + label_w
        for j, key in enumerate(("stock", "fast")):
            hi = r[key][1]
            rect(x0, y + j * 16, x0 + max(4, hi / axis_max * bar_w), y + j * 16 + 12, C[key], r=3)
        lo, hi = r["speedup"]
        sp = f"{lo:.1f}x" if f"{lo:.1f}" == f"{hi:.1f}" else f"{lo:.1f}–{hi:.1f}x"
        text(right, y + 14, sp, "bold", 22, C["text"], anchor="rm")
        y += 37
    text(cx, y + 4, "Median on each model's default GPU, from MODELS.md", "regular", 17, C["muted"])

    img.resize((W, H), Image.LANCZOS).save(HERE / "social-preview.png", optimize=True)


if __name__ == "__main__":
    main()

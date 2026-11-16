"""
make_screenshots.py - turn the console output of every script into terminal-style PNG images
for the report and the presentation.

It reads the text files in logs/ (created by run_all.py) and writes:
    screenshots/full/<script>.png                 the whole output of one script
    screenshots/sections/<script>__NN_<title>.png one image per table / section

Run:  python run_all.py            (once, to produce logs/)
      python make_screenshots.py
"""
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
LOGS, OUT = BASE / "logs", BASE / "screenshots"
FONT_SIZE, PAD, BAR = 18, 22, 40
BG, FG, BAR_BG, ACCENT, DIM = "#0C0C0C", "#DADADA", "#2B2B2B", "#6CB6FF", "#8A8A8A"


def font(size=FONT_SIZE, bold=False):
    for name in (["consolab.ttf", "courbd.ttf"] if bold else ["consola.ttf", "cour.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


F, FB = font(), font(bold=True)
CW = F.getlength("M")
LH = int(FONT_SIZE * 1.35)


def render(lines, title, path):
    lines = [l.rstrip("\n").expandtabs(4) for l in lines]
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return
    width = int(max(max(len(l) for l in lines), len(title) + 12, 60) * CW) + 2 * PAD
    height = BAR + 2 * PAD + LH * len(lines)
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, width, BAR], fill=BAR_BG)
    for i, c in enumerate(["#FF5F56", "#FFBD2E", "#27C93F"]):
        d.ellipse([14 + i * 22, 13, 28 + i * 22, 27], fill=c)
    d.text((90, 9), title, font=F, fill=DIM)
    y = BAR + PAD
    for l in lines:
        s = l.strip()
        heading = re.fullmatch(r"(CONCLUSIONS?|NOTE|DECISION.*|STEP .*|FINAL MODEL.*)", s) is not None
        if s.startswith("---") or s.startswith("===") or heading:
            d.text((PAD, y), l, font=FB, fill=ACCENT)
        elif s.startswith("[saved]"):
            d.text((PAD, y), l, font=F, fill=DIM)
        else:
            d.text((PAD, y), l, font=F, fill=FG)
        y += LH
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def sections(lines):
    """Split the output into blocks that start with '--- title', 'CONCLUSION...' or a dashed rule."""
    blocks, title, cur = [], "start", []
    for l in lines:
        s = l.strip()
        m = re.match(r"^--- (.+)$", s)
        new = m.group(1) if m else (s if re.match(r"^(CONCLUSIONS?|NOTE)$", s) else None)
        if s.startswith("Repeated hold-out"):
            new = "Repeated hold-out 100 splits"
        if s.startswith("Classification report"):
            new = "Classification report and confusion matrix"
        if new:
            blocks.append((title, cur))
            title, cur = new, [l]
        else:
            cur.append(l)
    blocks.append((title, cur))
    return [(t, [x for x in b if not x.strip().startswith("[saved]")]) for t, b in blocks if any(x.strip() for x in b)]


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:50]


def main():
    if not LOGS.exists() or not list(LOGS.glob("*.txt")):
        raise SystemExit("logs/ is empty - run  python run_all.py  first.")
    n = 0
    for log in sorted(LOGS.glob("*.txt")):
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        title = f"python {log.stem}.py"
        render(lines, title, OUT / "full" / f"{log.stem}.png")
        n += 1
        for i, (t, block) in enumerate(sections(lines), 1):
            render(block, f"{title}   -   {t}", OUT / "sections" / f"{log.stem}__{i:02d}_{slug(t)}.png")
            n += 1
        print(f"  {log.stem}")
    print(f"\n{n} images written to screenshots/")


if __name__ == "__main__":
    main()

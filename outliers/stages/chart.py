"""Animated streak-climb chart — the channel's signature stat visual.

Reads facts.json `comparison` (their_number vs second_best) and renders
content/<slug>/chart.mp4: a gold line climbing to the subject's number while
the runner-up's record sits as a gray dashed line near the floor. Generic:
works for any streak/gap video. Run standalone:

    python -m stages.chart <slug> [dur_seconds]
"""
import json, math, subprocess, sys, tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

GOLD = (250, 199, 117)
DIMGOLD = (74, 60, 36)
WHITE = (245, 245, 243)
GRAY = (125, 124, 118)
BG = (11, 11, 11)


def _ease(t: float) -> float:
    return 1 - (1 - t) ** 3


def run(out: Path, cfg: dict, dur: float = 3.5):
    if (out / "chart.mp4").exists():
        return
    facts = json.loads((out / "facts.json").read_text(encoding="utf-8"))
    c = facts["comparison"]
    target = float(c["their_number"])
    rec = float(c["second_best"]) if c.get("second_best") else None
    v = cfg["video"]; W, H, fps = v["width"], v["height"], v["fps"]
    big = ImageFont.truetype(v["caption_font"], 230)
    mid = ImageFont.truetype(v["caption_font"], 58)
    sml = ImageFont.truetype(v["caption_font"], 40)

    x0, x1 = W * 0.10, W * 0.90
    ybase, ytop = H * 0.80, H * 0.50
    ymax = target * 1.06
    n = int(dur * fps)
    climb_end = 0.80          # fraction of clip spent climbing; rest holds
    cross_frame = None

    tmp = Path(tempfile.mkdtemp(prefix="chart_"))
    for i in range(n):
        p = _ease(min(1.0, (i / max(n - 1, 1)) / climb_end))
        val = target * p
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)

        # runner-up record: dashed gray floor line (label drawn last, on top)
        ry = None
        if rec:
            ry = ybase - (ybase - ytop) * (rec / ymax)
            x = x0
            while x < x1:
                d.line([(x, ry), (min(x + 26, x1), ry)], fill=GRAY, width=3)
                x += 42
            if cross_frame is None and val >= rec:
                cross_frame = i

        # the climb: concave-up area + bright line + leading dot
        pts = []
        K = 72
        for k in range(K + 1):
            s = k / K
            fx = x0 + (x1 - x0) * s * p
            fy = ybase - (ybase - ytop) * ((target * (s ** 1.18) * p) / ymax)
            pts.append((fx, fy))
        d.polygon(pts + [(pts[-1][0], ybase), (x0, ybase)], fill=DIMGOLD)
        d.line(pts, fill=GOLD, width=6, joint="curve")
        d.ellipse([pts[-1][0] - 11, pts[-1][1] - 11,
                   pts[-1][0] + 11, pts[-1][1] + 11], fill=GOLD)

        if rec:
            hot = cross_frame is not None and (i - cross_frame) < fps * 0.45
            d.text((x0, ry - 58), f"{int(rec)}  {c.get('second_label', '')}".strip(),
                   font=sml, fill=WHITE if hot else GRAY)

        # counter
        d.text((W / 2, H * 0.315), f"{int(round(val))}", font=big, fill=GOLD, anchor="mm")
        d.text((W / 2, H * 0.405), c.get("their_label", ""), font=mid, fill=WHITE, anchor="mm")

        img.save(tmp / f"f{i:04d}.png")

    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps),
                    "-i", str(tmp / "f%04d.png"), "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", str(out / "chart.mp4")], check=True)
    (out / "chart.json").write_text(json.dumps({"dur": dur}), encoding="utf-8")
    print(f"chart: {int(target)} vs {int(rec) if rec else '—'} -> {out/'chart.mp4'} ({dur}s)")


if __name__ == "__main__":
    import yaml
    slug = sys.argv[1]
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 3.5
    run(Path("content") / slug, yaml.safe_load(open("config.yaml")), dur)

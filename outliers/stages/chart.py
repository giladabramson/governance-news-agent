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

    # histogram: runner-up bar (gray, left) vs subject bar (gold, right) rising
    GRAYBAR = (92, 91, 85)
    bw, gap = W * 0.20, W * 0.10
    lx0, rx0 = W / 2 - gap / 2 - bw, W / 2 + gap / 2
    ybase, bar_top = H * 0.74, H * 0.48    # chart area sits clear of the title block
    ymax = target * 1.08
    hgt = lambda v: (ybase - bar_top) * (v / ymax)
    n = int(dur * fps)
    cross_frame = None

    milestones = facts.get("chart_milestones", [])
    tmp = Path(tempfile.mkdtemp(prefix="chart_"))
    for i in range(n):
        val = target * _ease(min(1.0, (i / max(n - 1, 1)) / 0.85))
        rec_val = (rec or 0) * min(1.0, i / max(n * 0.22, 1))
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)

        # faint gridlines so the canvas reads as a chart, not a void
        for gfrac in (0.25, 0.5, 0.75, 1.0):
            gy = ybase - (ybase - bar_top) * gfrac
            d.line([(W * 0.08, gy), (W * 0.92, gy)], fill=(30, 30, 29), width=2)

        # counter + label up top
        d.text((W / 2, H * 0.245), f"{int(round(val))}", font=big, fill=GOLD, anchor="mm")
        d.text((W / 2, H * 0.335), c.get("their_label", ""), font=mid, fill=WHITE, anchor="mm")

        # milestone lines (e.g. "107 consecutive finals") — labels drawn after
        # the bars so the rising bar never punches through the text
        m_labels = []
        for m in milestones:
            my = ybase - (ybase - bar_top) * (float(m["value"]) / ymax)
            x = W * 0.08
            while x < W * 0.92:
                d.line([(x, my), (min(x + 18, W * 0.92), my)], fill=(64, 63, 58), width=2)
                x += 30
            near = val >= float(m["value"]) and (val - float(m["value"])) < target * 0.10
            ly = my + 34 if my - 40 < bar_top + 10 else my - 34
            # left side of the chart, where nothing rises into it
            m_labels.append(((W * 0.08 + 6, ly), m.get("label", str(m["value"])),
                             WHITE if near else GRAY))

        if rec:
            if cross_frame is None and val >= rec:
                cross_frame = i
            hot = cross_frame is not None and (i - cross_frame) < fps * 0.45
            d.rounded_rectangle([lx0, ybase - hgt(rec_val), lx0 + bw, ybase],
                                radius=14, fill=GRAYBAR)
            d.text((lx0 + bw / 2, ybase - hgt(rec_val) - 40), f"{int(rec)}",
                   font=sml, fill=WHITE if hot else GRAY, anchor="mm")

        d.rounded_rectangle([rx0, ybase - max(hgt(val), 8), rx0 + bw, ybase],
                            radius=14, fill=GOLD)

        for pos, txt, col in m_labels:
            d.text(pos, txt, font=sml, fill=col, anchor="lm")

        if rec and c.get("second_label"):
            d.text((W / 2, ybase + 44), f"{int(rec)} = {c['second_label']}",
                   font=sml, fill=GRAY, anchor="mm")

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

import json, re, requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

COMMONS = "https://commons.wikimedia.org/w/api.php"
OK_LICENSES = ("pd", "cc0", "cc-by", "cc by", "public domain")


def commons_search(q: str, ua: str, n: int = 4) -> list[dict]:
    r = requests.get(COMMONS, headers={"User-Agent": ua}, params={
        "action": "query", "generator": "search", "gsrsearch": q, "gsrnamespace": 6,
        "gsrlimit": n, "prop": "imageinfo", "iiprop": "url|extmetadata",
        "iiurlwidth": 1600, "format": "json"}, timeout=30).json()
    hits = []
    for p in r.get("query", {}).get("pages", {}).values():
        info = p["imageinfo"][0]
        url = info.get("thumburl", "")
        lic = info.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "").lower()
        # Filter on the source file's extension, not the thumb URL — thumburls carry
        # utm query params, and Commons happily thumbnails PDFs/TIFs we don't want.
        title = p.get("title", "").lower()
        if url and title.endswith((".jpg", ".jpeg", ".png")) and any(k in lic for k in OK_LICENSES):
            hits.append({"url": url, "license": lic, "page": p.get("title", "")})
    return hits


def _card(facts: dict, path: Path, cfg: dict):
    W, H = cfg["video"]["width"], cfg["video"]["height"]
    font = cfg["video"]["caption_font"]
    img = Image.new("RGB", (W, H), (11, 11, 11)); d = ImageDraw.Draw(img)
    big, mid, small = (ImageFont.truetype(font, s) for s in (220, 64, 48))
    c, kind = facts["comparison"], facts.get("comparison_type", "streak")
    gold, white, gray = (250, 199, 117), (255, 255, 255), (136, 135, 128)
    if kind == "gap":
        d.text((W / 2, H * 0.34), c["their_number"], font=big, fill=gold, anchor="mm")
        d.text((W / 2, H * 0.44), c["their_label"], font=mid, fill=white, anchor="mm")
        d.text((W / 2, H * 0.60), c["second_best"], font=big, fill=gray, anchor="mm")
        d.text((W / 2, H * 0.70), c["second_label"], font=mid, fill=gray, anchor="mm")
    elif kind == "streak":
        d.text((W / 2, H * 0.42), c["their_number"], font=big, fill=gold, anchor="mm")
        d.text((W / 2, H * 0.54), c["their_label"], font=mid, fill=white, anchor="mm")
    else:  # uniqueness
        d.multiline_text((W / 2, H * 0.5), c["gap_description"], font=mid, fill=gold,
                         anchor="mm", align="center", spacing=16)
    img.save(path)


def run(out: Path, cfg: dict):
    if (out / "shots.json").exists():
        return
    facts = json.loads((out / "facts.json").read_text(encoding="utf-8"))
    ua = cfg["commons"]["user_agent"]
    imgdir = out / "images"; imgdir.mkdir(exist_ok=True)
    paths, credits = [], []
    for q in facts["image_queries"]:
        for hit in commons_search(q, ua):
            p = imgdir / f"{len(paths):02d}.jpg"
            p.write_bytes(requests.get(hit["url"], headers={"User-Agent": ua}, timeout=60).content)
            paths.append(str(p)); credits.append(hit)
            if len(paths) >= 6: break
        if len(paths) >= 6: break
    if not paths:
        raise SystemExit("no usable Commons images — add image_queries in facts.json")
    (out / "credits.json").write_text(json.dumps(credits, indent=1), encoding="utf-8")
    card = imgdir / "card.png"; _card(facts, card, cfg)

    duration = float((out / "duration.txt").read_text(encoding="utf-8"))
    words = json.loads((out / "words.json").read_text(encoding="utf-8"))
    script = json.loads((out / "script.json").read_text(encoding="utf-8"))

    # The card shows only while the stat itself is spoken — the last sentence
    # of the number section — not the whole section. A 12s static card reads
    # like a slide; ~4s reads like a beat.
    n_before = len((script["hook"] + " " + script["story"]).split())
    num_words = script["number"].split()
    sents = re.split(r"(?<=[.!?])\s+", script["number"].strip())
    lead = len(num_words) - len(sents[-1].split()) if sents else 0
    card_start = words[min(n_before + lead, len(words) - 1)]["s"]
    sent_end = words[min(n_before + len(num_words) - 1, len(words) - 1)]["e"]
    card_end = min(max(sent_end, card_start + 3.5), duration)

    # Photo shots: ~2.1s cuts, cycling the pool with alternating camera moves.
    MOVES = ["zin", "panr", "zout", "panl"]
    shots, idx = [], 0

    def add_span(t0, t1):
        nonlocal idx
        n = max(1, round((t1 - t0) / 2.1))
        seg = (t1 - t0) / n
        for k in range(n):
            shots.append({"file": paths[idx % len(paths)],
                          "start": t0 + k * seg, "end": t0 + (k + 1) * seg,
                          "move": MOVES[idx % len(MOVES)]})
            idx += 1

    add_span(0.0, card_start)
    shots.append({"file": str(card), "start": card_start, "end": card_end, "static": True})
    if duration - card_end > 0.05:
        add_span(card_end, duration)
    (out / "shots.json").write_text(json.dumps(shots, indent=1), encoding="utf-8")
    print(f"visuals: {len(paths)} images + card, {len(shots)} shots, card {card_end-card_start:.1f}s")

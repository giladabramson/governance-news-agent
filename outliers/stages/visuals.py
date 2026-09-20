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
    # Commons full-text search matches file descriptions too, which drags in
    # unrelated images (a darts trophy for "Scrabble tournament"). Require the
    # TITLE to share at least one real word with the query.
    qtok = [t for t in re.findall(r"[a-z0-9]+", q.lower()) if len(t) > 3]
    hits = []
    for p in r.get("query", {}).get("pages", {}).values():
        info = p["imageinfo"][0]
        url = info.get("thumburl", "")
        lic = info.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "").lower()
        # Filter on the source file's extension, not the thumb URL — thumburls carry
        # utm query params, and Commons happily thumbnails PDFs/TIFs we don't want.
        title = p.get("title", "").lower()
        if (url and title.endswith((".jpg", ".jpeg", ".png"))
                and any(k in lic for k in OK_LICENSES)
                and (not qtok or any(t in title for t in qtok))):
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
    paths, credits, seen = [], [], set()
    for q in facts["image_queries"]:
        for hit in commons_search(q, ua, n=6):
            # Archives upload burst shots seconds apart whose titles differ only
            # in a serial number — digit-stripped title dedupes those sessions.
            key = "".join(c for c in hit["page"] if not c.isdigit())
            if key in seen:
                continue
            p = imgdir / f"{len(paths):02d}.jpg"
            p.write_bytes(requests.get(hit["url"], headers={"User-Agent": ua}, timeout=60).content)
            paths.append(str(p)); credits.append(hit); seen.add(key)
            if len(paths) >= 16: break
        if len(paths) >= 16: break
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

    # Cue images: when the narration says the word, show the thing.
    # facts.json: image_cues = [{match: [words], query: str, max: int}]
    _n = lambda t: "".join(c for c in t.lower() if c.isalnum())
    fixed = [{"file": str(card), "start": card_start, "end": card_end, "static": True}]
    for ci, cue in enumerate(facts.get("image_cues", [])):
        queries = cue["query"] if isinstance(cue["query"], list) else [cue["query"]]
        cue_paths = []
        for q in queries:
            for hit in commons_search(q, ua):
                key = "".join(c for c in hit["page"] if not c.isdigit())
                if key in seen:
                    continue
                p = imgdir / f"cue_{ci:02d}_{len(cue_paths)}.jpg"
                p.write_bytes(requests.get(hit["url"], headers={"User-Agent": ua}, timeout=60).content)
                cue_paths.append(str(p)); credits.append(hit); seen.add(key)
                break  # one image per query — variety over depth
            if len(cue_paths) >= cue.get("max", 2):
                break
        if not cue_paths:
            print(f"visuals: cue {queries[0]!r} — no Commons hits, skipped"); continue
        placed, last_t = 0, -9.0
        for w in words:
            if placed >= cue.get("max", 2):
                break
            if _n(w["w"]) in cue["match"] and w["s"] - last_t > 4.0:
                t0, t1 = max(0.0, w["s"] - 0.15), min(max(0.0, w["s"] - 0.15) + 2.2, duration)
                if all(t1 <= f["start"] or t0 >= f["end"] for f in fixed):
                    # rotate the cue's images so no file shows twice
                    fixed.append({"file": cue_paths[placed % len(cue_paths)],
                                  "start": t0, "end": t1, "move": "zin"})
                    placed += 1; last_t = w["s"]
    (out / "credits.json").write_text(json.dumps(credits, indent=1), encoding="utf-8")
    fixed.sort(key=lambda s: s["start"])

    # Fill the gaps with the pool: ~2.1s cuts, alternating camera moves.
    MOVES = ["zin", "panr", "zout", "panl"]
    shots, idx = [], 0

    def add_span(t0, t1):
        nonlocal idx
        if t1 - t0 < 0.25:
            # tiny sliver — stretch the previous shot instead of flashing one
            if shots and abs(shots[-1]["end"] - t0) < 0.01:
                shots[-1]["end"] = t1
            return
        n = max(1, round((t1 - t0) / 2.1))
        seg = (t1 - t0) / n
        for k in range(n):
            if idx == len(paths):
                print("visuals: pool exhausted — images will repeat; add image_queries")
            shots.append({"file": paths[idx % len(paths)],
                          "start": t0 + k * seg, "end": t0 + (k + 1) * seg,
                          "move": MOVES[idx % len(MOVES)]})
            idx += 1

    t = 0.0
    for f in fixed:
        add_span(t, f["start"])
        shots.append(f)
        t = f["end"]
    add_span(t, duration)
    shots.sort(key=lambda s: s["start"])
    (out / "shots.json").write_text(json.dumps(shots, indent=1), encoding="utf-8")
    ncues = len(fixed) - 1
    print(f"visuals: {len(paths)} pool + {ncues} cue shots + card, {len(shots)} total")

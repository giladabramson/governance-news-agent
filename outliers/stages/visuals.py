import difflib, json, re, requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def _is_damaged(path: Path) -> bool:
    """Reject archival scans with big pure-white gaps (torn/missing regions).
    >6% near-white pixels reliably flags the rectangle artifacts; clean B&W
    photos sit under ~4%."""
    try:
        im = Image.open(path).convert("L").resize((160, 160))
        px = im.getdata()
        return sum(1 for p in px if p > 250) / len(px) > 0.045
    except Exception:
        return False


def _sentence_ends(full_text: str, words: list) -> list:
    """Times where script sentences end, aligned onto heard words (script is
    the authority on punctuation, same approach as captions.py)."""
    _n = lambda t: re.sub(r"[^a-z0-9']", "", t.lower())
    sw = full_text.split()
    ends = [bool(re.search(r"[.!?][\"']?$", x)) for x in sw]
    sm = difflib.SequenceMatcher(None, [_n(x) for x in sw],
                                 [_n(x["w"]) for x in words], autojunk=False)
    flag = [w["w"].rstrip('"\'').endswith((".", "!", "?")) for w in words]
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in ("equal", "replace"):
            for d in range(min(i2 - i1, j2 - j1)):
                if ends[i1 + d]:
                    flag[j1 + d] = True
    return [w["e"] for w, f in zip(words, flag) if f]

COMMONS = "https://commons.wikimedia.org/w/api.php"
OK_LICENSES = ("pd", "cc0", "cc-by", "cc by", "public domain")


def commons_search(q: str, ua: str, n: int = 4) -> list[dict]:
    r = requests.get(COMMONS, headers={"User-Agent": ua}, params={
        "action": "query", "generator": "search", "gsrsearch": q, "gsrnamespace": 6,
        "gsrlimit": n, "prop": "imageinfo", "iiprop": "url|extmetadata",
        "iiurlwidth": 1600, "format": "json"}, timeout=30).json()
    # Commons full-text search matches file descriptions too, which drags in
    # unrelated images (a darts trophy for "Scrabble tournament"; a social-club
    # BUILDING for "Scrabble club"). Two gates: the title must share a word
    # with the query, and the query's FIRST word — by convention the topic
    # noun ("Scrabble ...", "draughts ...", "mainframe ...") — must appear in
    # the title or the file's Commons categories.
    qtok = [t for t in re.findall(r"[a-z0-9]+", q.lower()) if len(t) > 3]
    hits = []
    for p in r.get("query", {}).get("pages", {}).values():
        info = p["imageinfo"][0]
        url = info.get("thumburl", "")
        md = info.get("extmetadata", {})
        lic = md.get("LicenseShortName", {}).get("value", "").lower()
        cats = md.get("Categories", {}).get("value", "").lower()
        # Filter on the source file's extension, not the thumb URL — thumburls carry
        # utm query params, and Commons happily thumbnails PDFs/TIFs we don't want.
        title = p.get("title", "").lower()
        if (url and title.endswith((".jpg", ".jpeg", ".png"))
                and any(k in lic for k in OK_LICENSES)
                and (not qtok or any(t in title for t in qtok))
                and (not qtok or qtok[0] in title + " " + cats)):
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

    # Animated stat chart (stages/chart.py output): if present, it plays over
    # the hook's final sentence — the counter climbs while the stat is spoken.
    chart_fixed = None
    if (out / "chart.mp4").exists():
        chart_fixed = float(json.loads((out / "chart.json").read_text())["dur"]) \
            if (out / "chart.json").exists() else 3.5

    # Cues download FIRST and claim their images — otherwise a cue whose photo
    # also matches a pool query gets deduped away and silently never fires.
    cue_claimed = set()
    cue_downloads = []
    for ci, cue in enumerate(facts.get("image_cues", [])):
        queries = cue["query"] if isinstance(cue["query"], list) else [cue["query"]]
        crop = cue.get("crop")
        cue_paths = []
        for q in queries:
            q = q.lstrip("!")   # cues always take one hit; accept '!' anyway
            for hit in commons_search(q, ua):
                key = " ".join("".join(c for c in hit["page"] if not c.isdigit()).split()) + (crop or "")
                if key in seen:
                    continue
                p = imgdir / f"cue_{ci:02d}_{len(cue_paths)}.jpg"
                p.write_bytes(requests.get(hit["url"], headers={"User-Agent": ua}, timeout=60).content)
                if crop:
                    img = Image.open(p); W0, H0 = img.size
                    box = ((0.25, 0.06, 0.75, 0.60) if crop == "tight"
                           else (0.14, 0.03, 0.86, 0.80))
                    img.crop((int(W0*box[0]), int(H0*box[1]),
                              int(W0*box[2]), int(H0*box[3]))).save(p)
                cue_paths.append(str(p)); credits.append(hit); seen.add(key)
                # also claim the un-cropped identity so the pool skips it
                cue_claimed.add(" ".join("".join(c for c in hit["page"] if not c.isdigit()).split()))
                break
            if len(cue_paths) >= cue.get("max", 2):
                break
        cue_downloads.append((cue, cue_paths))

    # Push-ins are reserved for the subject: a zoom onto a face reads as "this
    # is who we're talking about". Images from queries naming the subject get
    # zin; everything else pans or pulls back.
    subj_tokens = {t for t in re.findall(r"[a-z]+", facts["name"].lower()) if len(t) > 3}
    subj_flags = []
    _kless = lambda t: " ".join("".join(c for c in t if not c.isdigit()).split())
    _kfull = lambda t: " ".join(t.split())
    for q in facts["image_queries"]:
        # "!" prefix = precision query: take exactly ONE hit (fuzzy Commons
        # ranking loves to hand precise queries a stranger in slot two)
        cap = 1 if q.startswith("!") else 2
        q = q.lstrip("!")
        per_q = 0  # cap per query so no single photo series dominates the pool
        q_subj = any(t in q.lower() for t in subj_tokens)
        for hit in commons_search(q, ua, n=6):
            if per_q >= cap:
                break
            # Burst dedupe (digit-stripped title) applies to NEUTRAL images.
            # For the subject, distinct frames of the same session are wanted
            # — a subject-starved video should show them, not filler.
            if _kfull(hit["page"]) in seen or _kless(hit["page"]) in cue_claimed:
                continue
            if not q_subj and _kless(hit["page"]) in seen:
                continue
            p = imgdir / f"{len(paths):02d}.jpg"
            p.write_bytes(requests.get(hit["url"], headers={"User-Agent": ua}, timeout=60).content)
            if _is_damaged(p):
                p.unlink(missing_ok=True)
                print(f"visuals: skipped damaged scan {hit['page'][:50]}")
                seen.add(_kfull(hit["page"])); continue
            paths.append(str(p)); credits.append(hit)
            seen.add(_kfull(hit["page"])); seen.add(_kless(hit["page"]) if not q_subj else _kfull(hit["page"]))
            per_q += 1
            subj_flags.append(q_subj)
            if len(paths) >= 24: break
        if len(paths) >= 24: break
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
    # Number card is opt-in (config video.number_card) — Gilad cut it: a text
    # slide, even drifting, reads as a frozen frame. Verdicts live in narration.
    fixed = []
    if cfg["video"].get("number_card", False):
        fixed.append({"file": str(card), "start": card_start, "end": card_end, "static": True})
    if chart_fixed:
        # Place the chart where the stat is SPOKEN: facts.chart_match lists
        # trigger words; fallback is the hook's final sentence.
        _nw = lambda t: "".join(ch for ch in t.lower() if ch.isalnum())
        t0 = None
        for trig in facts.get("chart_match", []):
            hitw = next((w for w in words if _nw(w["w"]) == trig), None)
            if hitw:
                t0 = max(0.0, hitw["s"] - 0.3); break
        if t0 is None:
            hook_words = script["hook"].split()
            sents = re.split(r"(?<=[.!?])\s+", script["hook"].strip())
            lead = len(hook_words) - len(sents[-1].split()) if len(sents) > 1 else 0
            t0 = max(0.0, words[min(lead, len(words) - 1)]["s"] - 0.1)
        fixed.append({"file": str(out / "chart.mp4"), "start": t0,
                      "end": min(t0 + chart_fixed, duration), "video": True})
    for cue, cue_paths in cue_downloads:
        if not cue_paths:
            print(f"visuals: cue {cue['match']} — no Commons hits, skipped"); continue
        placed, last_t = 0, -9.0
        for w in words:
            if placed >= cue.get("max", 2):
                break
            if _n(w["w"]) in cue["match"] and w["s"] - last_t > 4.0:
                t0, t1 = max(0.0, w["s"] - 0.15), min(max(0.0, w["s"] - 0.15) + 2.2, duration)
                # Clip against already-fixed shots (card, earlier cues) instead
                # of rejecting on any overlap — a cue 0.1s inside the card
                # should start at the card's edge, not vanish.
                for f in sorted(fixed, key=lambda x: x["start"]):
                    if t0 < f["end"] and t1 > f["start"]:
                        if f["start"] - t0 >= t1 - f["end"]:
                            t1 = f["start"]
                        else:
                            t0 = f["end"]
                if t1 - t0 >= 1.2:
                    # rotate the cue's images so no file shows twice
                    fixed.append({"file": cue_paths[placed % len(cue_paths)],
                                  "start": t0, "end": t1, "move": "zin"})
                    placed += 1; last_t = w["s"]
    (out / "credits.json").write_text(json.dumps(credits, indent=1), encoding="utf-8")
    fixed.sort(key=lambda s: s["start"])

    # Fill the gaps with the pool: ~2.1s cuts, alternating camera moves.
    # Never repeat an image: if the pool can't cover every 2.1s slot, ration —
    # fewer, slightly longer cuts per gap instead of a second showing.
    SUBJ_MOVES = ["zin", "panr"]
    NEUTRAL_MOVES = ["panr", "zout", "panl"]
    sent_ends = _sentence_ends(script["full_text"], words)
    gaps, t = [], 0.0
    for f in fixed:
        if f["start"] - t >= 0.25:
            gaps.append((t, f["start"]))
        t = f["end"]
    if duration - t >= 0.25:
        gaps.append((t, duration))

    # Allocate images to gaps PROPORTIONALLY — sequential greed starves the
    # final gap into one endless shot at the video's climax.
    wants = [max(1, round((t1 - t0) / 2.1)) for t0, t1 in gaps]
    if sum(wants) > len(paths) and gaps:
        scale = len(paths) / sum(wants)
        alloc = [max(1, int(w * scale)) for w in wants]
        rema = sorted(range(len(wants)), key=lambda i: wants[i] * scale - int(wants[i] * scale), reverse=True)
        k = 0
        while sum(alloc) < len(paths) and k < len(rema):
            alloc[rema[k]] += 1; k += 1
        while sum(alloc) > len(paths):
            cands = [i for i in range(len(alloc)) if alloc[i] > 1]
            if not cands:
                break   # more gaps than images — repeats unavoidable
            j = max(cands, key=lambda i: alloc[i] / max(wants[i], 1))
            alloc[j] -= 1
    else:
        alloc = wants

    shots, idx = [dict(f) for f in fixed], 0
    for gi, (t0, t1) in enumerate(gaps):
        n = alloc[gi]
        if idx + n > len(paths):
            print("visuals: pool exhausted — images will repeat; add image_queries")
        # Cut on sentence ends when possible: snap each uniform cut point to
        # the nearest sentence boundary within 0.8s (keeping shots >= 1.0s).
        seg = (t1 - t0) / n
        bounds = [t0]
        for k in range(1, n):
            ideal = t0 + k * seg
            b = ideal
            if sent_ends:
                snap = min(sent_ends, key=lambda s: abs(s - ideal)) + 0.05
                if (abs(snap - ideal) <= 0.85 and snap - bounds[-1] >= 1.0
                        and t1 - snap >= 1.0):
                    b = snap
            bounds.append(b)
        bounds.append(t1)
        for k in range(n):
            j = idx % len(paths)
            subj = subj_flags[j] if j < len(subj_flags) else False
            move = (SUBJ_MOVES if subj else NEUTRAL_MOVES)[idx % (2 if subj else 3)]
            shots.append({"file": paths[j],
                          "start": bounds[k], "end": bounds[k + 1],
                          "move": move})
            idx += 1
    shots.sort(key=lambda s: s["start"])
    (out / "shots.json").write_text(json.dumps(shots, indent=1), encoding="utf-8")
    ncards = sum(1 for f in fixed if f.get("static"))
    print(f"visuals: {len(paths)} pool + {len(fixed)-ncards} cue shots"
          + (" + card" if ncards else "") + f", {len(shots)} total")

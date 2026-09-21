import difflib, json, re
from pathlib import Path
from faster_whisper import WhisperModel


def _ts(sec: float) -> str:
    h = int(sec // 3600); m = int(sec % 3600 // 60); s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def run(out: Path, cfg: dict):
    if (out / "captions.ass").exists():
        return
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(out / "voice.mp3"), word_timestamps=True)
    words = [{"w": w.word.strip(), "s": w.start, "e": w.end}
             for seg in segments for w in seg.words]
    if not words:
        raise SystemExit("captions: whisper returned no words — check voice.mp3")

    # Whisper splits decimals/thousands across tokens ("99" + ".94.") — merge
    # them back so numbers are whole words for captions, cues and alignment.
    merged = []
    for w in words:
        t = w["w"].strip()
        if (merged and re.fullmatch(r"[.,]\d+[.,]?", t)
                and merged[-1]["w"].strip().rstrip(".,").isdigit()):
            merged[-1]["w"] = merged[-1]["w"].strip() + t
            merged[-1]["e"] = w["e"]
        else:
            merged.append(dict(w))
    words = merged
    (out / "words.json").write_text(json.dumps(words, indent=1), encoding="utf-8")

    W, H = cfg["video"]["width"], cfg["video"]["height"]
    header = f"""[Script Info]
PlayResX: {W}
PlayResY: {H}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Inter,88,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,0,5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    # Group up to 3 words, never across a sentence boundary. Whisper's own
    # punctuation is unreliable (it heard the beat after "In checkers." as a
    # comma), so sentence ends come from the SCRIPT: align heard words to
    # script words and carry the script's boundary flags over.
    _n = lambda t: re.sub(r"[^a-z0-9']", "", t.lower())
    sw = json.loads((out / "script.json").read_text(encoding="utf-8"))["full_text"].split()
    ends = [bool(re.search(r"[.!?][\"']?$", x)) for x in sw]
    sm = difflib.SequenceMatcher(None, [_n(x) for x in sw],
                                 [_n(x["w"]) for x in words], autojunk=False)
    flag = [w["w"].rstrip('"\'').endswith((".", "!", "?")) for w in words]
    # The script is also the authority on how NUMBERS look: Whisper writes
    # "9994"/"9,994" where the script says "99.94"/"9994" — display the
    # script's token whenever a 1:1 aligned pair involves digits.
    disp_over = {}
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in ("equal", "replace"):
            for d in range(min(i2 - i1, j2 - j1)):
                if ends[i1 + d]:
                    flag[j1 + d] = True
                if (i2 - i1) == (j2 - j1) \
                        and any(ch.isdigit() for ch in sw[i1 + d]) \
                        and sw[i1 + d].strip('.,') != words[j1 + d]["w"].strip('.,'):
                    disp_over[j1 + d] = sw[i1 + d]

    groups, grp = [], []
    for i, w in enumerate(words):
        if i in disp_over:
            w = dict(w, w=disp_over[i])
        grp.append(w)
        if len(grp) == 3 or flag[i]:
            groups.append(grp); grp = []
    if grp:
        groups.append(grp)

    # Spoken-word captions carry no punctuation — a period on screen reads as
    # text, not speech. Grouping above still uses the raw words for boundaries.
    def _disp(w: str) -> str:
        return w.strip().strip('.,;:!?"“”«»')

    # Each line extends to the NEXT word's start — a pause mid-group must not
    # blank the caption and re-show the same text (flicker Gilad spotted).
    # Across group boundaries, hold up to 0.8s into the pause.
    lines = []
    for gi, grp in enumerate(groups):
        nxt_start = groups[gi + 1][0]["s"] if gi + 1 < len(groups) else None
        for j, cur in enumerate(grp):
            if j + 1 < len(grp):
                end = max(cur["e"], grp[j + 1]["s"])
            elif nxt_start is not None:
                end = max(cur["e"], min(nxt_start, cur["e"] + 0.8))
            else:
                end = cur["e"]
            txt = " ".join(
                (r"{\c&H00FFFF&}" + _disp(g["w"]) + r"{\c&HFFFFFF&}") if k == j else _disp(g["w"])
                for k, g in enumerate(grp))
            lines.append(f"Dialogue: 0,{_ts(cur['s'])},{_ts(end)},Cap,,0,0,0,,{txt}")
    (out / "captions.ass").write_text(header + "\n".join(lines), encoding="utf-8")
    (out / "duration.txt").write_text(str(words[-1]["e"] + 0.8), encoding="utf-8")
    print(f"captions: {len(words)} words, {words[-1]['e']:.1f}s")

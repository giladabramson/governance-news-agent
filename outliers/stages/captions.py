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
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in ("equal", "replace"):
            for d in range(min(i2 - i1, j2 - j1)):
                if ends[i1 + d]:
                    flag[j1 + d] = True

    groups, grp = [], []
    for i, w in enumerate(words):
        grp.append(w)
        if len(grp) == 3 or flag[i]:
            groups.append(grp); grp = []
    if grp:
        groups.append(grp)

    # Spoken-word captions carry no punctuation — a period on screen reads as
    # text, not speech. Grouping above still uses the raw words for boundaries.
    def _disp(w: str) -> str:
        return w.strip().strip('.,;:!?"“”«»')

    lines = []
    for grp in groups:
        for j, cur in enumerate(grp):
            txt = " ".join(
                (r"{\c&H00FFFF&}" + _disp(g["w"]) + r"{\c&HFFFFFF&}") if k == j else _disp(g["w"])
                for k, g in enumerate(grp))
            lines.append(f"Dialogue: 0,{_ts(cur['s'])},{_ts(cur['e'])},Cap,,0,0,0,,{txt}")
    (out / "captions.ass").write_text(header + "\n".join(lines), encoding="utf-8")
    (out / "duration.txt").write_text(str(words[-1]["e"] + 0.8), encoding="utf-8")
    print(f"captions: {len(words)} words, {words[-1]['e']:.1f}s")

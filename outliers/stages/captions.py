import json
from pathlib import Path
from faster_whisper import WhisperModel


def _ts(sec: float) -> str:
    h = int(sec // 3600); m = int(sec % 3600 // 60); s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def run(out: Path, cfg: dict):
    if (out / "captions.ass").exists():
        return
    model = WhisperModel("small", compute_type="int8")
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
    lines = []
    for i in range(0, len(words), 3):
        grp = words[i:i + 3]
        for j, cur in enumerate(grp):
            txt = " ".join(
                (r"{\c&H00FFFF&}" + g["w"] + r"{\c&HFFFFFF&}") if k == j else g["w"]
                for k, g in enumerate(grp))
            lines.append(f"Dialogue: 0,{_ts(cur['s'])},{_ts(cur['e'])},Cap,,0,0,0,,{txt}")
    (out / "captions.ass").write_text(header + "\n".join(lines), encoding="utf-8")
    (out / "duration.txt").write_text(str(words[-1]["e"] + 0.8), encoding="utf-8")
    print(f"captions: {len(words)} words, {words[-1]['e']:.1f}s")

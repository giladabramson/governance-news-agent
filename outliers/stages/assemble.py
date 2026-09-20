import json, subprocess, random
from pathlib import Path

# ffmpeg's filtergraph parser treats ':' as an option separator, so a Windows
# absolute path ("C:/...") breaks the ass= filter. Everything we reference lives
# under the project root, so pass it a relative POSIX path instead — no colon.
def _filter_path(p: Path) -> str:
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return p.as_posix().replace(":", r"\:")


def _kb(src, dur, W, H, fps, zoom_in=True):
    frames = int(dur * fps)
    z = f"1+0.12*on/{frames}" if zoom_in else f"1.12-0.12*on/{frames}"
    return (f"[{src}]scale={W*2}:{H*2}:force_original_aspect_ratio=increase,"
            f"crop={W*2}:{H*2},zoompan=z='{z}':d={frames}:s={W}x{H}:fps={fps}"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)',setsar=1")


def run(out: Path, cfg: dict):
    if (out / "final.mp4").exists():
        return
    v = cfg["video"]; W, H, fps = v["width"], v["height"], v["fps"]
    shots = json.loads((out / "shots.json").read_text(encoding="utf-8"))
    # Music is optional: an empty assets/music renders voice-only rather than failing.
    tracks = sorted(Path("assets/music").glob("*.mp3"))
    music = random.choice(tracks) if tracks else None

    inputs, filters, labels = [], [], []
    for i, s in enumerate(shots):
        dur = s["end"] - s["start"]
        inputs += ["-loop", "1", "-t", f"{dur:.3f}", "-i", s["file"]]
        if s.get("static"):
            filters.append(f"[{i}:v]scale={W}:{H},setsar=1,fps={fps}[v{i}]")
        else:
            filters.append(_kb(f"{i}:v", dur, W, H, fps, zoom_in=(i % 2 == 0)) + f"[v{i}]")
        labels.append(f"[v{i}]")
    n = len(shots)
    ass = _filter_path(out / "captions.ass")
    fontsdir = _filter_path(Path(v["caption_font"]).parent)
    filters.append("".join(labels) + f"concat=n={n}:v=1:a=0[vc]")
    filters.append(f"[vc]ass={ass}:fontsdir={fontsdir}[vout]")
    if music:
        filters.append(f"[{n}:a]aformat=channel_layouts=stereo[voice]")
        filters.append(f"[{n+1}:a]volume={v['music_volume_db']}dB,aformat=channel_layouts=stereo[mus]")
        filters.append("[mus][voice]sidechaincompress=threshold=0.05:ratio=6:attack=20:release=400[musd]")
        filters.append("[voice][musd]amix=inputs=2:duration=first:dropout_transition=2[aout]")
        audio_in = ["-stream_loop", "-1", "-i", str(music)]
    else:
        filters.append(f"[{n}:a]aformat=channel_layouts=stereo[aout]")
        audio_in = []
        print("assemble: no music in assets/music — rendering voice-only")

    cmd = ["ffmpeg", "-y", *inputs, "-i", str(out / "voice.mp3"), *audio_in,
           "-filter_complex", ";".join(filters),
           "-map", "[vout]", "-map", "[aout]", "-shortest",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out / "final.mp4")]
    subprocess.run(cmd, check=True)
    print(f"done -> {out/'final.mp4'}")

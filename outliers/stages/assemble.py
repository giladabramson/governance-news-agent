import json, re, subprocess, random
from pathlib import Path


def _versions(out: Path):
    """Existing cuts, sorted: [(1, v1.mp4), (2, v2.mp4), ...]"""
    vers = []
    for p in out.glob("v*.mp4"):
        m = re.fullmatch(r"v(\d+)", p.stem)
        if m:
            vers.append((int(m.group(1)), p))
    return sorted(vers)

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
    # Versioned output: each render that follows changed inputs (or --redo
    # assemble) writes the next vN.mp4; earlier cuts are never overwritten.
    vers = _versions(out)
    deps = [out / f for f in ("voice.mp3", "captions.ass", "shots.json", "script.json")]
    deps += [p for p in [Path("assets/sfx/hook.mp3")] if p.exists()]
    if vers and not cfg.get("_force_assemble"):
        newest_dep = max(d.stat().st_mtime for d in deps if d.exists())
        if vers[-1][1].stat().st_mtime >= newest_dep:
            return
    target = out / f"v{vers[-1][0] + 1 if vers else 1}.mp4"
    v = cfg["video"]; W, H, fps = v["width"], v["height"], v["fps"]
    shots = json.loads((out / "shots.json").read_text(encoding="utf-8"))
    # Music is optional: an empty assets/music renders voice-only rather than failing.
    tracks = sorted(Path("assets/music").glob("*.mp3"))
    music = random.choice(tracks) if tracks else None

    inputs, filters, labels = [], [], []
    for i, s in enumerate(shots):
        dur = s["end"] - s["start"]
        if s.get("static"):
            inputs += ["-loop", "1", "-t", f"{dur:.3f}", "-i", s["file"]]
            filters.append(f"[{i}:v]scale={W}:{H},setsar=1,fps={fps}[v{i}]")
        else:
            # Single-frame input: zoompan's d=<frames> generates the whole clip
            # from one frame. Looping the input first would make zoompan expand
            # EVERY input frame by d, ballooning the stream to hours.
            inputs += ["-i", s["file"]]
            filters.append(_kb(f"{i}:v", dur, W, H, fps, zoom_in=(i % 2 == 0)) + f"[v{i}]")
        labels.append(f"[v{i}]")
    n = len(shots)

    # Captions sit mid-screen, which collides with the number card's text.
    # While the card is up, push caption lines to the lower third ({\an2} +
    # MarginV) so the card owns the center. Derived file; original untouched.
    ass_use = out / "captions.ass"
    card = next((s for s in shots if s.get("static")), None)
    if card:
        def _sec(ts):
            h, m, sec = ts.split(":")
            return int(h) * 3600 + int(m) * 60 + float(sec)
        fixed = []
        for ln in (out / "captions.ass").read_text(encoding="utf-8").splitlines():
            if ln.startswith("Dialogue:"):
                parts = ln.split(",", 9)
                if card["start"] - 0.05 <= _sec(parts[1]) < card["end"]:
                    parts[7] = "380"
                    parts[9] = r"{\an2}" + parts[9]
                    ln = ",".join(parts)
            fixed.append(ln)
        ass_use = out / "captions_burn.ass"
        ass_use.write_text("\n".join(fixed), encoding="utf-8")

    ass = _filter_path(ass_use)
    fontsdir = _filter_path(Path(v["caption_font"]).parent)
    filters.append("".join(labels) + f"concat=n={n}:v=1:a=0[vc]")
    filters.append(f"[vc]ass={ass}:fontsdir={fontsdir}[vout]")

    # Audio inputs after the stills: voice at [n], then optional music / hook SFX.
    audio_in, ai = [], n + 1
    music_idx = sfx_idx = None
    if music:
        audio_in += ["-stream_loop", "-1", "-i", str(music)]
        music_idx = ai; ai += 1
    else:
        print("assemble: no music in assets/music — rendering voice-only")
    sfx, hook_end = Path("assets/sfx/hook.mp3"), None
    if sfx.exists():
        try:
            script = json.loads((out / "script.json").read_text(encoding="utf-8"))
            words = json.loads((out / "words.json").read_text(encoding="utf-8"))
            hook_end = words[min(len(script["hook"].split()), len(words)) - 1]["e"]
        except Exception:
            pass
    if hook_end is not None:
        audio_in += ["-i", str(sfx)]
        sfx_idx = ai; ai += 1

    filters.append(f"[{n}:a]aformat=channel_layouts=stereo[voice]")
    vsrc = "voice"
    if sfx_idx is not None:
        ms = int(hook_end * 1000)
        filters.append(f"[{sfx_idx}:a]adelay={ms}|{ms},aformat=channel_layouts=stereo,volume=-8dB[sfx]")
        filters.append("[voice][sfx]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[vmix]")
        vsrc = "vmix"
    if music_idx is not None:
        filters.append(f"[{music_idx}:a]volume={v['music_volume_db']}dB,aformat=channel_layouts=stereo[mus]")
        filters.append(f"[mus][{vsrc}]sidechaincompress=threshold=0.05:ratio=6:attack=20:release=400[musd]")
        filters.append(f"[{vsrc}][musd]amix=inputs=2:duration=first:dropout_transition=2[aout]")
    else:
        filters.append(f"[{vsrc}]anull[aout]")

    cmd = ["ffmpeg", "-y", *inputs, "-i", str(out / "voice.mp3"), *audio_in,
           "-filter_complex", ";".join(filters),
           "-map", "[vout]", "-map", "[aout]", "-shortest",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(target)]
    subprocess.run(cmd, check=True)
    print(f"done -> {target}")

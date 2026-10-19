import sys, argparse, yaml
from pathlib import Path
from stages import tts, captions, visuals, assemble


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--voice", action="store_true", help="allow ElevenLabs call (spends credits)")
    ap.add_argument("--draft", action="store_true",
                    help="free preview: Windows SAPI voice instead of ElevenLabs")
    ap.add_argument("--until", choices=["tts", "captions", "visuals", "assemble"], default="assemble")
    ap.add_argument("--redo", nargs="*", default=[], help="stages to force, e.g. --redo visuals assemble")
    a = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))
    out = Path("content") / a.slug
    assert (out / "script.json").exists(), "run /script first"

    outputs = {"tts": ["voice.mp3"], "captions": ["captions.ass", "words.json", "duration.txt"],
               "visuals": ["shots.json", "credits.json"], "assemble": []}
    for st in a.redo:
        for f in outputs[st]:
            (out / f).unlink(missing_ok=True)
    # Old cuts (vN.mp4) are never deleted — --redo assemble just forces a new one.
    if "assemble" in a.redo:
        cfg["_force_assemble"] = True

    if a.draft:
        cfg["_draft"] = True
    elif not a.voice and not (out / "voice.mp3").exists():
        sys.exit("voice.mp3 missing — rerun with --voice to spend credits (or --draft for a free preview)")

    for name, mod in [("tts", tts), ("captions", captions), ("visuals", visuals), ("assemble", assemble)]:
        print(f"--- {name}"); mod.run(out, cfg)
        if name == a.until: break


if __name__ == "__main__":
    main()

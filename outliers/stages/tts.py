import json, os, subprocess
from pathlib import Path
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings


def _trim_lead(mp3: Path):
    """Tighten silences without touching the voice: cut the leading dead air
    (pure hook lag) and clamp any internal pause to <=0.5s. Runs before
    captions, so Whisper times against the edited audio — nothing desyncs.
    Pause complaints are an ffmpeg edit, never a paid re-generation."""
    t = mp3.with_suffix(".trim.mp3")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(mp3), "-af",
                    "silenceremove=start_periods=1:start_threshold=-40dB:start_silence=0.08"
                    ":stop_periods=-1:stop_threshold=-40dB:stop_duration=0.5,"
                    "areverse,"
                    "silenceremove=start_periods=1:start_threshold=-55dB:start_silence=0.35,"
                    "areverse",
                    "-c:a", "libmp3lame", "-q:a", "2", str(t)], check=True)
    t.replace(mp3)


def run(out: Path, cfg: dict):
    mp3, script = out / "voice.mp3", out / "script.json"
    if mp3.exists() and mp3.stat().st_mtime >= script.stat().st_mtime:
        return
    load_dotenv()
    sd = json.loads(script.read_text(encoding="utf-8"))
    if all(k in sd for k in ("hook", "story", "number", "closer")):
        # Audible pauses between sections. ElevenLabs renders <break /> as
        # silence, so Whisper's word timings stay aligned with the script text.
        # Inside the hook, a beat before its final sentence makes the punch
        # land ("...every six years. (beat) In checkers.").
        import re
        hook_sents = re.split(r"(?<=[.!?])\s+", sd["hook"].strip())
        hook = ' <break time="0.25s" /> '.join(
            [" ".join(hook_sents[:-1]), hook_sents[-1]]) if len(hook_sents) > 1 else sd["hook"]
        text = (f'{hook} <break time="0.25s" /> {sd["story"]} '
                f'<break time="0.15s" /> {sd["number"]} '
                f'<break time="0.2s" /> {sd["closer"]}')
    else:
        text = sd["full_text"]
    if cfg.get("_draft"):
        # Free preview voice via Windows SAPI — no credits, real word timings
        # for captions/visuals. Swapped for ElevenLabs on the final render.
        import re, subprocess, tempfile
        plain = re.sub(r"<break[^>]*/>", " . ", text)
        tf = Path(tempfile.gettempdir()) / "outliers_draft_text.txt"
        tf.write_text(plain, encoding="utf-8-sig")
        wav = mp3.with_suffix(".draft.wav")
        ps = ("Add-Type -AssemblyName System.Speech; "
              "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              "$s.Rate = 2; "
              f"$s.SetOutputToWaveFile('{wav}'); "
              f"$s.Speak([IO.File]::ReadAllText('{tf}')); $s.Dispose()")
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(wav),
                        "-c:a", "libmp3lame", "-q:a", "4", str(mp3)], check=True)
        wav.unlink(missing_ok=True)
        _trim_lead(mp3)
        print(f"tts: DRAFT voice (Windows SAPI, free) -> {mp3}")
        return

    t = cfg["tts"]
    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    audio = client.text_to_speech.convert(
        voice_id=os.environ["ELEVENLABS_VOICE_ID"],
        text=text,
        model_id=t["model_id"],
        output_format="mp3_44100_128",
        voice_settings=VoiceSettings(stability=t["stability"],
                                     similarity_boost=t["similarity_boost"],
                                     speed=t.get("speed", 1.0)),
    )
    with open(mp3, "wb") as f:
        for chunk in audio:
            f.write(chunk)
    _trim_lead(mp3)
    print(f"tts: {len(text)} chars -> {mp3}")

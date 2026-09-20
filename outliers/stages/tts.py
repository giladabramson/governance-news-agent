import json, os
from pathlib import Path
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings


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
        hook = ' <break time="0.4s" /> '.join(
            [" ".join(hook_sents[:-1]), hook_sents[-1]]) if len(hook_sents) > 1 else sd["hook"]
        text = (f'{hook} <break time="0.25s" /> {sd["story"]} '
                f'<break time="0.15s" /> {sd["number"]} '
                f'<break time="0.2s" /> {sd["closer"]}')
    else:
        text = sd["full_text"]
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
    print(f"tts: {len(text)} chars -> {mp3}")

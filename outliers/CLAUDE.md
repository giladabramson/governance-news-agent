# Outliers Shorts

YouTube Shorts about people who dominated their field to a statistically absurd degree
(Nigel Richards, Marion Tinsley, Don Bradman). Tone: calm, dry, matter-of-fact. Numbers do the work.

## Workflow per video
1. `/research "Full Name"` → writes `content/<slug>/facts.json`. I review and may edit.
2. `/script` (run inside that folder context) → writes `script.json`. I edit the hook by hand.
3. `python render.py <slug> --voice` → voice, captions, visuals, and a new cut `v<N>.mp4`
   (versions are never overwritten; `--redo assemble` forces the next one).
4. `/meta` → `meta.json` (title, description with sources, tags). I upload manually.

## Hard rules
- Never put a number in a script that is not in facts.json with a source_url.
- If sources disagree on a number, facts.json must say so and the script must attribute ("by most counts", "per the Chinook team").
- Scripts are 120–160 words. Hook ≤ 15 words. No "did you know", no "legendary", no "incredible".
- Do not run tts.py yourself; it spends credits. I run render.py.
- Do not edit files under content/*/ except facts.json, script.json, meta.json.

## Slugs
lowercase, hyphens: "Marion Tinsley" → marion-tinsley

## Environment (this machine)
- Python lives in the shared workspace venv: `c:\dev\.venv\Scripts\python.exe`. Run commands
  from `c:\dev\outliers\` using that interpreter, or activate with `..\.venv\Scripts\Activate.ps1`.
- `ffmpeg.exe` / `ffprobe.exe` are in `c:\dev\.venv\Scripts\` — on PATH once the venv is active.
- This folder is part of the `c:\dev` git repo, not a repo of its own.

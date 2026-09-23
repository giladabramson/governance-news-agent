# Outliers Shorts

YouTube Shorts about people who dominated their field to a statistically absurd degree
(Nigel Richards, Marion Tinsley, Don Bradman). Tone: calm, dry, matter-of-fact. Numbers do the work.

## Workflow per video — three human gates, then hands-off
1. `/research "Full Name"` → `content/<slug>/facts.json`.
   **GATE 1: Gilad reviews the facts in chat. Iterate until he approves. Do not proceed.**
2. `/script` → `script.json`.
   **GATE 2: Gilad reviews the script in chat (enumerated: hook/story/number/closer). Iterate. Do not render.**
3. Download the image pool + cues and show Gilad EVERY picture (contact sheet).
   **GATE 3: he vetoes/keeps images in chat; adjust queries and re-show until approved.**
4. `python render.py <slug> --draft` → free preview cut with the Windows SAPI voice
   standing in for ElevenLabs. Gilad watches the draft (images in context, captions,
   cuts). Iterate visuals here — drafts cost nothing.
5. Only after draft approval: `python render.py <slug> --voice --redo tts captions visuals assemble`
   (spends credits) → verify (transcript match, zero repeats, frame extraction) →
   `/meta` → Gilad uploads.
   Versions are never overwritten (`v<N>.mp4`); `--redo assemble` forces the next one.

## Hard rules
- Never put a number in a script that is not in facts.json with a source_url.
- If sources disagree on a number, facts.json must say so and the script must attribute ("by most counts", "per the Chinook team").
- Scripts are 120–160 words. Hook ≤ 15 words. No "did you know", no "legendary", no "incredible".
- Every script establishes the sport early: one line on what the game is and what the
  key number means (e.g. cricket's average, squash's format). Never assume the viewer knows.
- Do not run tts.py yourself; it spends credits. I run render.py.
- Do not edit files under content/*/ except facts.json, script.json, meta.json.

## Slugs
lowercase, hyphens: "Marion Tinsley" → marion-tinsley

## Environment (this machine)
- Python lives in the shared workspace venv: `c:\dev\.venv\Scripts\python.exe`. Run commands
  from `c:\dev\outliers\` using that interpreter, or activate with `..\.venv\Scripts\Activate.ps1`.
- `ffmpeg.exe` / `ffprobe.exe` are in `c:\dev\.venv\Scripts\` — on PATH once the venv is active.
- This folder is part of the `c:\dev` git repo, not a repo of its own.

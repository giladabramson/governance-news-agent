Write the narration script for the video in the folder given in $ARGUMENTS
(or the most recently created content/ folder if none given).

Read `facts.json` first. Use ONLY numbers that appear there.

Structure, exactly four parts, 120–160 words total:
1. hook — 1 sentence, ≤ 15 words. The most unbelievable fact, stated flatly.
2. story — 3–4 sentences. Who they were, what they did, the story_beat.
3. number — 2 sentences. The comparison. If comparison_type is "uniqueness", make it
   a statement of what nobody else has done instead of a numeric gap.
4. closer — 1 sentence, dry and memorable. No call to action.

Voice: calm, matter-of-fact, slightly amused. Short sentences. No "incredible", "amazing",
"legendary", "did you know". If a number is disputed in facts.json, attribute it in the
script ("by most counts", "per the Chinook team").

Write THREE alternative hooks first, print them, then pick the strongest for the script.

Write `content/<slug>/script.json`:
{
  "hook": "...", "story": "...", "number": "...", "closer": "...",
  "full_text": "<the four parts joined with spaces>",
  "hook_alternatives": ["...", "...", "..."],
  "word_count": 0
}

Then print full_text and the word count. Stop. I will edit before rendering.

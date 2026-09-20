Build a source card for the person named in $ARGUMENTS.

Steps:
1. Fetch the English Wikipedia article. Fetch 2–3 more primary-ish sources (an obituary,
   a long-form profile, an official record page). Use web search / fetch tools.
2. Extract the facts that make this person a statistical outlier. Prefer numbers with a
   direct comparison to the #2 in history.
3. Where sources disagree, keep BOTH numbers in one fact with a `note`, and mark confidence: medium.
4. Pick the single best anecdote (story_beat) and its source.
5. Decide comparison_type: gap | streak | uniqueness.
6. Write `content/<slug>/facts.json` following the schema in CLAUDE.md / spec section 4.
   Every fact must have a real source_url you actually fetched. Never invent a URL.
7. Print a 5-line summary: the hook candidate, the comparison, and any disputed number.

Accuracy checks before you write the file:
- Pin every anecdote to a specific event and year. If a famous quote or moment is commonly
  misattributed to the wrong match/season, say which one it actually belongs to in the note.
- A number that only one source carries is confidence: medium at best, never high.

Do not write the script. Stop after facts.json.

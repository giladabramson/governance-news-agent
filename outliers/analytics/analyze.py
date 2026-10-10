import pandas as pd

# per-video totals (from Table data.csv), mapped to our subjects
rows = [
 ("Karelin",  "The Takedown of an Unbeatable Monster",            "Oct 3", 44, 1701, 5.588,  1, 88,  3.41),
 ("Marciano", "The tiny boxer who never lost a fight",            "Oct 5", 40, 1467, 7.3847, 3, 494, 2.43),
 ("Gretzky",  "The Most Generous Athlete is also the Best",       "Sep 27",39, 1288, 3.7779, 2, 55,  0.0),
 ("Moses",    "The Craziest Olympic Champion is a Geek",          "Sep 22",40, 1159, 3.2887, 1, 81,  1.23),
 ("Paris",    "The Most Impressive Mom in Sports History",        "Sep 23",38, 1159, 3.4087, 1, 62,  0.0),
 ("Bradman",  "They Had to Change the Rules To Stop Him",         "Sep 29",46, 1119, 2.9845, 1, 70,  1.43),
 ("Tinsley",  "The world champion who lost once every six years", "Sep 20",48, 893,  3.1213, 0, 90,  3.33),
 ("Vergeer",  "She didn't lose a tennis match for ten years",     "Sep 21",48, 662,  1.9693, 1, 162, 1.23),
 ("Frankel",  "This Horse is the GOAT",                           "Oct 1", 37, 228,  0.3493, 0, 40,  2.5),
 ("Richards", "World champion in 3 languages. He speaks 1.",      "Sep 20",46, 176,  0.3684, 0, 750, 1.07),
 ("Khan",     "The Longest Win Streak in Sports History",         "Sep 25",49, 143,  0.3653, 0, 86,  3.49),
]
df = pd.DataFrame(rows, columns=["subject","title","pub","dur","views","wh","subs","impr","ctr"])
df["avg_sec"]   = df["wh"]*3600/df["views"]
df["pct_view"]  = (df["avg_sec"]/df["dur"]*100).round(1)
df["subs_per1k"]= (df["subs"]/df["views"]*1000).round(2)

print("=== BY RETENTION (avg % of video watched — the quality signal) ===")
print(df.sort_values("pct_view", ascending=False)[["subject","pub","views","pct_view","avg_sec","subs","subs_per1k"]].to_string(index=False))
print(f"\nchannel totals: {df['views'].sum()} views, {df['subs'].sum()} subs from these 11")
print(f"median retention: {df['pct_view'].median():.0f}%  |  median views: {int(df['views'].median())}")

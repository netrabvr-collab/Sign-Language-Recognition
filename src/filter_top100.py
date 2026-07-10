import json
from collections import Counter

with open("WLASL_v0.3.json") as f:
    data = json.load(f)

gloss_counts = [(entry["gloss"], len(entry["instances"])) for entry in data]
gloss_counts.sort(key=lambda x: -x[1])

top_100_glosses = set(g for g, _ in gloss_counts[:100])
print("Top 100 glosses selected. Sample:", list(top_100_glosses)[:10])

filtered = [entry for entry in data if entry["gloss"] in top_100_glosses]

with open("WLASL100.json","w") as f:
    json.dump(filtered,f)

total_videos = sum(len(entry["instances"]) for entry in filtered)
print(f"Filtered dataset: {len(filtered)} glosses, {total_videos} total video instances")
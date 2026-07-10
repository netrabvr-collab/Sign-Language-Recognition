import json
import os
import shutil

with open("WLASL100.json") as f:
    data = json.load(f)

src_dir = "videos"
dst_dir = os.path.join("..", "..", "data", "raw")  

os.makedirs(dst_dir, exist_ok=True)

copied = 0
missing = 0

for entry in data:
    gloss = entry["gloss"]
    gloss_dir = os.path.join(dst_dir, gloss)
    os.makedirs(gloss_dir, exist_ok=True)

    for inst in entry["instances"]:
        video_id = inst["video_id"]
        src_path = os.path.join(src_dir, f"{video_id}.mp4")
        dst_path = os.path.join(gloss_dir, f"{video_id}.mp4")

        if os.path.exists(src_path):
            shutil.copyfile(src_path, dst_path)
            copied += 1
        else:
            missing += 1

print(f"Copied: {copied}, Missing: {missing}")
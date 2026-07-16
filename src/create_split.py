import os
import glob
import json
import random

random.seed(42)  # reproducibility

def collect_samples(processed_dir):
    samples = []  # list of (filepath, label)
    words = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    for word in words:
        files = glob.glob(os.path.join(processed_dir, word, "*.npy"))
        for f in files:
            samples.append((f, word))
    return samples

def stratified_split(samples, train_ratio=0.7, val_ratio=0.15):
    # group by label
    by_label = {}
    for path, label in samples:
        by_label.setdefault(label, []).append(path)

    train, val, test = [], [], []

    for label, paths in by_label.items():
        random.shuffle(paths)
        n = len(paths)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n > 2 else 0

        train += [(p, label) for p in paths[:n_train]]
        val += [(p, label) for p in paths[n_train:n_train + n_val]]
        test += [(p, label) for p in paths[n_train + n_val:]]

    return train, val, test

if __name__ == "__main__":
    self_samples = collect_samples("data/processed_self")
    wlasl_samples = collect_samples("data/processed")

    all_samples = self_samples + wlasl_samples
    print(f"Total samples: {len(all_samples)}")

    train, val, test = stratified_split(all_samples)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    split_data = {
        "train": train,
        "val": val,
        "test": test
    }

    os.makedirs("data/splits", exist_ok=True)
    with open("data/splits/split.json", "w") as f:
        json.dump(split_data, f, indent=2)

    print("Saved data/splits/split.json")
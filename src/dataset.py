import torch
from torch.utils.data import Dataset
import numpy as np
import json

class SignLanguageDataset(Dataset):
    def __init__(self, split_json_path, split_name, label_to_idx):
        with open(split_json_path) as f:
            data = json.load(f)

        self.samples = data[split_name]
        self.label_to_idx = label_to_idx
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        filepath, label = self.samples[idx]
        sequence = np.load(filepath)
        sequence = torch.tensor(sequence, dtype=torch.float32)

        label_idx = self.label_to_idx[label]
        label_tensor = torch.tensor(label_idx, dtype = torch.long)

        return sequence, label_tensor
    

def build_label_mapping(split_json_path):
    """Scan all aplits to get every unique label, build a consistent word <-> index mapping"""
    with open(split_json_path) as f:
        data = json.load(f)
    
    all_labels = set()
    for split_name in ["train","val","test"]:
        for filepath,label in data[split_name]:
            all_labels.add(label)
    
    sorted_labels = sorted(all_labels)
    label_to_idx = {label:idx for idx, label in enumerate(sorted_labels)}
    idx_to_label = {idx: label for label,idx in label_to_idx.items()}

    return label_to_idx, idx_to_label

if __name__ == "__main__":
    label_to_idx, idx_to_label = build_label_mapping('data/splits/split.json')
    print(f"Number of classes : {len(label_to_idx)}")
    
    train_dataset = SignLanguageDataset("data/splits/split.json", "train", label_to_idx)
    print(f"Train dataset size: {len(train_dataset)}")

    sample_seq, sample_label = train_dataset[0]
    print(f"Sample sequence shape: {sample_seq.shape}")
    print(f"Sample label index: {sample_label.item()} -> word: {idx_to_label[sample_label.item()]}")
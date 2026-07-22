import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os

from model import SignLSTM
from dataset import SignLanguageDataset, build_label_mapping


#Config
SPLIT_JSON = 'data/splits/split_top30.json' 
BATCH_SIZE = 16
NUM_EPOCHS = 60
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.4
MODEL_SAVE_PATH = 'models/best_model_top30.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {device}")

#Data
label_to_idx,idx_to_label = build_label_mapping(SPLIT_JSON)
num_classes = len(label_to_idx)

train_dataset = SignLanguageDataset(SPLIT_JSON, "train", label_to_idx, augment=True)
val_dataset = SignLanguageDataset(SPLIT_JSON, 'val', label_to_idx, augment = False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train sample : {len(train_dataset)}, Val samples : {len(val_dataset)}, Classes : {num_classes}")

#model
model = SignLSTM(
    input_size=225,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=num_classes,
    dropout=DROPOUT
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)


def run_epoch(loader, training=True):
    model.train() if training else model.eval()

    total_loss = 0
    correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item() * sequences.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

if __name__ == "__main__":
    os.makedirs("models",exist_ok = True)
    best_val_acc = 0.0
    epochs_without_improvement = 0
    PATIENCE = 8

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(train_loader, training=True)
        val_loss, val_acc = run_epoch(val_loader, training=False)

        print(f"Epoch {epoch}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label,
                "val_acc": val_acc,
            }, MODEL_SAVE_PATH)
            print(f"  -> New best model saved (val_acc: {val_acc:.4f})")
        
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping : no improvement for {PATIENCE} epochs")
            break


    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
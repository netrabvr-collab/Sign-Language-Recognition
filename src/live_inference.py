import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque

from model import SignLSTM
from extract_landmarks import extract_landmarks_from_frame, normalize_sequence

mp_holistic = mp.solutions.holistic

MODEL_PATH = "models/best_model_top30.pt"
SEQUENCE_LENGTH = 40
CONFIDENCE_THRESHOLD = 0.6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load model ----
checkpoint = torch.load(MODEL_PATH, map_location=device)
label_to_idx = checkpoint["label_to_idx"]
idx_to_label = checkpoint["idx_to_label"]
num_classes = len(label_to_idx)

model = SignLSTM(input_size=225, hidden_size=64, num_layers=1, num_classes=num_classes, dropout=0.4)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print(f"Loaded model with val_acc: {checkpoint['val_acc']:.4f}")
print(f"Classes: {list(label_to_idx.keys())}")

# ---- Rolling frame buffer ----
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)


def predict_from_buffer():
    if len(frame_buffer) < SEQUENCE_LENGTH:
        return None, 0.0

    sequence = np.array(frame_buffer)  # (40, 225)
    sequence = normalize_sequence(sequence)

    tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 40, 225)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    predicted_word = idx_to_label[pred_idx.item()]
    return predicted_word, confidence.item()


def main():
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            vec = extract_landmarks_from_frame(results)
            frame_buffer.append(vec)

            predicted_word, confidence = predict_from_buffer()

            if predicted_word and confidence > CONFIDENCE_THRESHOLD:
                text = f"{predicted_word} ({confidence:.2f})"
                color = (0, 255, 0)
            else:
                text = "..." if predicted_word is None else f"low confidence ({confidence:.2f})"
                color = (0, 0, 255)

            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Live Sign Recognition", frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
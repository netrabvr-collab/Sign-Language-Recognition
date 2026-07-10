import cv2
import os

VOCAB_FILE = "data/vocab.txt"
OUTPUT_DIR = "data/raw_self"
TAKES_PER_WORD = 10
RECORD_SECONDS = 2
FPS = 20

with open(VOCAB_FILE) as f:
    words = [w.strip() for w in f if w.strip()]

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

for word in words:
    word_dir = os.path.join(OUTPUT_DIR, word)
    os.makedirs(word_dir, exist_ok=True)

    for take in range(1, TAKES_PER_WORD + 1):
        print(f"\nNext: '{word}' - take {take}/{TAKES_PER_WORD}")
        print("Press 'r' to start recording, 'q' to quit entirely.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, f"{word} ({take}/{TAKES_PER_WORD}) - press r", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Recorder", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                break
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        out_path = os.path.join(word_dir, f"{take}.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, FPS, (int(cap.get(3)), int(cap.get(4))))

        num_frames = RECORD_SECONDS * FPS
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            cv2.putText(frame, "RECORDING", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Recorder", frame)
            cv2.waitKey(1)

        writer.release()
        print(f"Saved {out_path}")

cap.release()
cv2.destroyAllWindows()
print("Done recording all words.")
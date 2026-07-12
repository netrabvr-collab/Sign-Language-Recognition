import cv2
import mediapipe as mp
import numpy as np
import os
import glob

mp_holistic = mp.solutions.holistic

SEQUENCE_LENGTH = 40  # fixed number of frames per sample


def get_coords(landmark_list, num_points):
    if landmark_list:
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in landmark_list.landmark]
        ).flatten()
    return np.zeros(num_points * 3)


def extract_landmarks_from_frame(results):
    pose = get_coords(results.pose_landmarks, 33)
    left_hand = get_coords(results.left_hand_landmarks, 21)
    right_hand = get_coords(results.right_hand_landmarks, 21)
    return np.concatenate([pose, left_hand, right_hand])


def normalize_sequence(sequence):
    """
    Normalize each frame relative to the pose's mid-hip/torso point and
    scale by shoulder width, so position/distance from camera don't matter.
    Pose landmarks: indices 0-32 in the flattened 99-length pose block.
    Landmark 11 = left shoulder, 12 = right shoulder (x,y,z each, 3 values apart).
    """
    normalized = sequence.copy()

    for i in range(len(sequence)):
        frame = sequence[i]
        pose = frame[:99].reshape(33, 3)

        left_shoulder = pose[11]
        right_shoulder = pose[12]
        center = (left_shoulder + right_shoulder) / 2  # reference point
        shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)

        if shoulder_dist < 1e-6:  # avoid divide-by-zero if pose not detected
            shoulder_dist = 1.0

        # Reshape whole frame into (num_points, 3), subtract center, divide by scale
        all_points = frame.reshape(-1, 3)
        all_points = (all_points - center) / shoulder_dist
        normalized[i] = all_points.flatten()

    return normalized


def pad_or_truncate(sequence, target_length=SEQUENCE_LENGTH):
    num_frames = len(sequence)
    feature_dim = sequence.shape[1]

    if num_frames == target_length:
        return sequence
    elif num_frames > target_length:
        # Sample evenly across the sequence instead of just cutting the end
        indices = np.linspace(0, num_frames - 1, target_length).astype(int)
        return sequence[indices]
    else:
        # Pad with zeros at the end
        pad_amount = target_length - num_frames
        padding = np.zeros((pad_amount, feature_dim))
        return np.concatenate([sequence, padding], axis=0)


def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            vec = extract_landmarks_from_frame(results)
            sequence.append(vec)

    cap.release()
    return np.array(sequence)


def process_one_video(video_path, save_path):
    if os.path.exists(save_path):
        return "skipped"

    sequence = extract_landmarks_from_video(video_path)
    if len(sequence) == 0:
        return "empty"

    sequence = normalize_sequence(sequence)
    sequence = pad_or_truncate(sequence)

    np.save(save_path, sequence)
    return "ok"


def process_dataset(raw_dir, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    words = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]

    stats = {"ok": 0, "skipped": 0, "empty": 0}

    for word in words:
        word_raw_dir = os.path.join(raw_dir, word)
        word_out_dir = os.path.join(processed_dir, word)
        os.makedirs(word_out_dir, exist_ok=True)

        video_files = glob.glob(os.path.join(word_raw_dir, "*.mp4"))

        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            save_path = os.path.join(word_out_dir, f"{video_name}.npy")

            result = process_one_video(video_path, save_path)
            stats[result] += 1

        print(f"Finished '{word}': {len(video_files)} videos processed so far. Running totals: {stats}")

    print(f"\nFINAL STATS: {stats}")


if __name__ == "__main__":
    print("Processing self-recorded dataset...")
    process_dataset("data/raw_self", "data/processed_self")

    print("\nProcessing WLASL dataset...")
    process_dataset("data/raw", "data/processed")
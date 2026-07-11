import mediapipe as mp
import cv2
import numpy as np
import os

mp_holistic = mp.solutions.holistic

def extract_landmarks_from_frames(results):
    def get_coords(landmark_list, num_points):
        if landmark_list:
            return np.array(
                [[lm.x,lm.y,lm.z] for lm in landmark_list.landmark]
            ).flatten()
        return np.zeros(num_points * 3)
    
    pose = get_coords(results.pose_landmarks, 33)
    left_hand = get_coords(results.left_hand_landmarks,21)
    right_hand = get_coords(results.right_hand_landmarks, 21)
    return np.concatenate([pose,left_hand, right_hand])

def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []
    with mp_holistic.Holistic(
        min_detection_confidence = 0.5, min_tracking_confidence = 0.5
    ) as holistic:
        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            vec = extract_landmarks_from_frames(results)
            sequence.append(vec)
    cap.release()
    return np.array(sequence)

def process_one_video(video_path, save_path):
    if os.path.exists(save_path):
        return "skipped"
    sequence = extract_landmarks_from_video(video_path)
    if len(sequence) == 0:
        return "empty"
    np.save(save_path,sequence)
    return "ok"

if __name__ == "__main__":
    test_video = "data/raw_self/hello/1.mp4"
    test_output = "data/processed_test.npy"

    print(f"Testing extraction on: {test_video}")
    result = process_one_video(test_video, test_output)
    print(f"Result: {result}")

    if result == "ok":
        loaded = np.load(test_output)
        print(f"Extracted sequence shape: {loaded.shape}")
        print(f"Expected:(num_frames, 258) - num_frames should be roughly {20*2} (FPS=20, 2 sec)")

import numpy as np
import matplotlib.pyplot as plt

self_sample = np.load("data/processed_self/hello/1.npy")
wlasl_sample = np.load("data/processed/book/07069.npy")

print(f"Self-recorded 'hello' shape: {self_sample.shape}")
print(f"WLASL 'book' shape: {wlasl_sample.shape}")

def get_right_wrist_trajectory(seq):
    right_hand_block = seq[:, 162:225].reshape(-1,21,3)
    wrist = right_hand_block[:,0,:]
    return wrist[:,0], wrist[:,1]

xs,ys = get_right_wrist_trajectory(wlasl_sample)
plt.plot(xs,ys,marker='o')
plt.gca().invert_yaxis()
plt.title("Right wrist trajectory: pre-recorded 'book'")
plt.savefig("sanity_check_trajectory.png")
print("Saved sanity_check_trajectory.png")

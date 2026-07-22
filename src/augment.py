import numpy as np


def rotate_sequence(sequence, max_angle_deg=10):
    """Rotate all landmarks around the origin (in the x-y plane) by a small random angle."""
    angle = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    rotated = sequence.copy()
    points = rotated.reshape(rotated.shape[0], -1, 3)  # (frames, num_points, 3)

    x = points[:, :, 0].copy()
    y = points[:, :, 1].copy()
    points[:, :, 0] = x * cos_a - y * sin_a
    points[:, :, 1] = x * sin_a + y * cos_a

    return points.reshape(sequence.shape)


def scale_sequence(sequence, scale_range=(0.9, 1.1)):
    """Scale all landmarks by a small random factor."""
    scale = np.random.uniform(*scale_range)
    return sequence * scale


def add_noise(sequence, noise_std=0.01):
    """Add small random jitter to every landmark coordinate."""
    noise = np.random.normal(0, noise_std, sequence.shape)
    return sequence + noise


def time_warp(sequence, warp_range=(0.85, 1.15)):
    """Slightly speed up or slow down the sequence, then resample back to original length."""
    original_len = sequence.shape[0]
    warp_factor = np.random.uniform(*warp_range)
    warped_len = max(2, int(original_len * warp_factor))

    indices = np.linspace(0, original_len - 1, warped_len)
    warped = sequence[np.round(indices).astype(int)]

    # Resample back to original length
    final_indices = np.linspace(0, warped_len - 1, original_len)
    resampled = warped[np.round(final_indices).astype(int)]

    return resampled


def augment_sequence(sequence):
    """Apply a random combination of augmentations to one sequence."""
    aug = sequence.copy()

    if np.random.rand() < 0.5:
        aug = rotate_sequence(aug)
    if np.random.rand() < 0.5:
        aug = scale_sequence(aug)
    if np.random.rand() < 0.7:
        aug = add_noise(aug)
    if np.random.rand() < 0.5:
        aug = time_warp(aug)

    return aug


if __name__ == "__main__":
    # Quick test
    sample = np.load("data/processed_self/hello/1.npy")
    print(f"Original shape: {sample.shape}")

    augmented = augment_sequence(sample)
    print(f"Augmented shape: {augmented.shape}")
    print(f"Max difference from original: {np.abs(sample - augmented).max():.4f}")
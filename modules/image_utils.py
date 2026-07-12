import numpy as np
from scipy.ndimage import gaussian_filter
from .constants import IMAGE_THRESHOLD, VOLTAGE


def image_to_pulses(
    img,
    intervals=0,
    binarize=True,
    threshold=IMAGE_THRESHOLD,
    dtype=np.uint8,
    mode="LINEAR",
    voltage=VOLTAGE,
):
    if mode == "LINEAR":
        pulse_trains = img.flatten()

        if binarize:
            pulse_trains = binarize_image(pulse_trains, threshold)

        zeros = np.zeros((1 + intervals) * len(pulse_trains), dtype=dtype)
        zeros[:: (1 + intervals)] = pulse_trains
        pulse_trains = zeros

        pulse_trains = pulse_trains.reshape(
            (img.shape[0], img.shape[1] * (1 + intervals))
        ).T

        return pulse_trains

    if mode == "ISOMETRIC":
        h, w = img.shape
        pulse_train_new = img.flatten()

        if binarize:
            pulse_trains = binarize_image(pulse_trains, threshold)

        pulse_train_new = (
            pulse_train_new.reshape(h // 2, 2, w // 2, 2)
            .transpose(0, 2, 1, 3)
            .reshape(-1)
        ).flatten()

        pulse_train_new = (pulse_train_new / 255) * voltage

        zeros = np.zeros((1 + intervals) * len(pulse_train_new), dtype=np.float64)
        zeros[:: (1 + intervals)] = pulse_train_new
        pulse_train_new = zeros

        pulse_train_new = pulse_train_new.reshape(
            (7 * img.shape[0], len(pulse_train_new) // (7 * img.shape[0]))
            # (img.shape[0], img.shape[1] * (1 + intervals))
        )

        return pulse_train_new


def center_surround(image, sigma=0.8):
    blur = gaussian_filter(image, sigma=sigma)

    Xcs = image - blur
    Xon = np.maximum(Xcs, 0)

    Xon = (Xon > 160).astype(np.uint8)

    return Xon


def percentil_strech(image, low=5, high=95, eps=1e-8):
    low_mask = np.percentile(image, low)
    high_mask = np.percentile(image, high)

    treated_image = (image - low_mask) / (high_mask - low_mask + eps)

    return np.clip(treated_image, 0, 1)


def gamma_scaling(image, gamma=0.7):
    return np.clip(image, 0, 1) ** gamma


def lateral_inibition(image, sigma=0.8, k=1.2):
    blur = gaussian_filter(image, sigma=sigma)

    final_image = image + k * (image - blur)

    final_image = (final_image > 1.04).astype(np.uint8)

    return np.clip(final_image, 0, 1)


def binarize_image(img, threshold=IMAGE_THRESHOLD):
    return (img > threshold).astype(np.ubyte)


def binarize_images(imgs, threshold=IMAGE_THRESHOLD):
    return [binarize_image(img, threshold) for img in imgs]


def downscale_image(img, shape):
    return img[:: img.shape[0] // shape[0], :: img.shape[1] // shape[1]]


def downscale_images(imgs, shape):
    return [downscale_image(img, shape) for img in imgs]

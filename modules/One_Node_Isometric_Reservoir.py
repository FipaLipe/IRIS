import numpy as np
from .IGT_Simulation import generate_single_out_current
from .image_utils import (
    image_to_pulses,
    center_surround,
    lateral_inibition,
    gamma_scaling,
    percentil_strech,
)
from .constants import (
    K,
    BETA,
    TAU,
    IMAGE_THRESHOLD,
    TIME_RESOLUTION,
)
import matplotlib.pyplot as plt


def run_reservoir(images, time_resolution=TIME_RESOLUTION, tranpose=False):
    reconstructed_images_reservoir = []
    reconstructed_images = []

    for i, img in enumerate(images):
        intervals = 1
        if tranpose:
            img_pulses = image_to_pulses(img.T, intervals, False, mode="ISOMETRIC")
        else:
            img_pulses = image_to_pulses(img, intervals, False, mode="ISOMETRIC")

        new_img = []

        for pulse in img_pulses:
            out_current = generate_single_out_current(
                pulse, K, BETA, TAU, TIME_RESOLUTION
            )
            new_img.append(out_current[-1])

        new_img = np.array(new_img)
        downscaled_img = (
            (np.array(images[i]) > IMAGE_THRESHOLD).astype(np.uint8)[::2, ::4].flatten()
        )

        reconstructed_images_reservoir.append(new_img)
        reconstructed_images.append(downscaled_img)

    return reconstructed_images_reservoir, reconstructed_images


def resize_images(images, time_resolution=TIME_RESOLUTION):
    reconstructed_images, _ = run_reservoir(images, time_resolution=time_resolution)

    reconstructed_images = (np.array(reconstructed_images)[:, :]).reshape(
        (len(images), 14, 14)
    )

    return reconstructed_images


def downscale_treatment(images, time_resolution=TIME_RESOLUTION, image_threshold=0.999):

    filtered_images = []
    for image in images:
        filtered_images.append(center_surround(image))
        # filtered_images.append(image)

    down_scale_form = resize_images(images, time_resolution=time_resolution)
    reservoir_image = resize_images(filtered_images, time_resolution=time_resolution)

    filtered_images = []
    for image_treated, image in zip(reservoir_image, down_scale_form):
        treated_image = gamma_scaling(percentil_strech(image_treated))
        # filtered_images.append(treated_image)

        contorn = (lateral_inibition(treated_image) > 0.8).astype(np.uint8)
        full_image = (image > 0).astype(np.uint8)

        # final_images = np.clip(1.1*full_image - contorn, 0, 1)
        # filtered_images.append((final_images > image_threshold).astype(np.uint8))
        filtered_images.append(np.clip(1.1 * full_image - contorn, 0, 1))

    return np.array(filtered_images)

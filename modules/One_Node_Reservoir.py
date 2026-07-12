import numpy as np
from .IGT_Simulation import generate_single_out_current
from .image_utils import image_to_pulses, downscale_images
from .constants import K, BETA, TAU, TIME_RESOLUTION, IMAGE_THRESHOLD

def reconstruct_images(images, max_current, k=K, beta=BETA, tau=TAU, time_resolution=TIME_RESOLUTION, image_threshold=IMAGE_THRESHOLD):
    reconstructed_images = []

    images = downscale_images(images, (14, 28))

    for img in images:
        img_pulses = image_to_pulses(img, intervals=1)
        img_pulses = img_pulses.T.reshape((img_pulses.size//8, 8))

        new_img = []
        
        for pulse in img_pulses:
            out_current = generate_single_out_current(pulse, k, beta, tau, time_resolution)
            new_img.append(out_current[-1] / max_current)

        new_img = np.array(new_img)

        reconstructed_images.append(new_img)
    
    return reconstructed_images
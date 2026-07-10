import numpy as np
from .constants import IMAGE_THRESHOLD

def image_to_pulses(img, intervals=0, binarize=True, threshold=IMAGE_THRESHOLD, dtype=np.uint8):
    pulse_trains = img.flatten()

    if binarize:
        pulse_trains = binarize_image(pulse_trains, threshold)

    zeros = np.zeros((1+intervals) * len(pulse_trains), dtype=dtype)
    zeros[::(1+intervals)] = pulse_trains
    pulse_trains = zeros

    pulse_trains = pulse_trains.reshape((img.shape[0], img.shape[1]*(1+intervals))).T

    return pulse_trains

def binarize_image(img, threshold=IMAGE_THRESHOLD):
    return (img > threshold).astype(np.ubyte)

def binarize_images(imgs, threshold=IMAGE_THRESHOLD):
    return [binarize_image(img, threshold) for img in imgs]

def downscale_image(img, shape):
    return img[::img.shape[0]//shape[0], ::img.shape[1]//shape[1]]

def downscale_images(imgs, shape):
    return [downscale_image(img, shape) for img in imgs]
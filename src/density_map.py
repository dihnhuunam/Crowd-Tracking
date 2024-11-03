#GENERAL
import numpy as np
import random

# PATH PROCESS
import os
from scipy.io import loadmat

# IMAGE PROCESSING
import cv2
from scipy.ndimage import gaussian_filter

# NEURAL NETWORK
import torch
from torch.utils.data import Dataset

def gen_density_map_gaussian(image, coords, sigma=5):
    img_zeros = np.zeros((image.shape[:2]), dtype=np.float32)
    for x_cor, y_cor in coords:
        img_zeros[int(y_cor), int(x_cor)] = 1

    density_map = gaussian_filter(img_zeros, sigma=sigma, truncate=5*5)

    return density_map

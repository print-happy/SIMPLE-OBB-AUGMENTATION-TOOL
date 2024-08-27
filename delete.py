import cv2
import numpy as np
import os
from tqdm import tqdm

image_dir = ''
image_range=

def delete_out_of_range_images(image_dir, start_range, end_range):
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            image_number = int(filename.split('.')[0])
            if not (start_range <= image_number <= end_range):
                os.remove(os.path.join(image_dir, filename))
                print(f"Deleted {filename}")

delete_out_of_range_images(image_dir, 1, image_range)


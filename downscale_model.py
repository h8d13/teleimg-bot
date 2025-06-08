import cv2
import numpy as np
from .base_model import BaseModel

class DownscaleModel(BaseModel):
    @staticmethod
    def process_image(image):
        # Get the dimensions of the original image
        original_height, original_width = image.shape[:2]

        # Calculate the dimensions of the new image
        new_width = int(original_width * 0.66)
        new_height = int(original_height * 0.66)

        # Downscale the image using INTER_AREA interpolation
        downscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        return downscaled

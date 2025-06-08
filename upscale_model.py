import cv2
import numpy as np
from .base_model import BaseModel

## CAN INTRODUCE SETTING FOR SCALE FACTOR

class UpscaleModel(BaseModel):
    @staticmethod
    def process_image(image):
        # Get the dimensions of the original image
        original_height, original_width = image.shape[:2]

        # Calculate the dimensions of the new image
        new_width = int(original_width * 1.33)
        new_height = int(original_height * 1.33)

        # Upscale the image using INTER_CUBIC interpolation
        upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        return upscaled


import cv2
import numpy as np
from .base_model import BaseModel

class SimpleContrastModel(BaseModel):
    @staticmethod
    def process_image(image, contrast=1.05, sharpness=1.08):
        # Convert image to float
        f = image.astype(float)
        
        # Compute the mean brightness
        mean = np.mean(f, axis=(0, 1))
        
        # Apply contrast adjustment
        contrasted = (f - mean) * contrast + mean
        
        # Clip values to valid range
        contrasted = np.clip(contrasted, 0, 255)
        
        # Convert back to 8-bit
        contrasted = contrasted.astype(np.uint8)
        
        # Create sharpening kernel
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]]) * (sharpness - 1)
        kernel[1, 1] += 1
        
        # Apply sharpening
        sharpened = cv2.filter2D(contrasted, -1, kernel)
        
        return sharpened
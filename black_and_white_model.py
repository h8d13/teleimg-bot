import cv2
import numpy as np
from .base_model import BaseModel

class BlackAndWhiteModel(BaseModel):
    @staticmethod
    def process_image(image, contrast=1.2,  clipLimit=1.2, tileGridSize=(2,2)):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        clahe_img = clahe.apply(gray)

        # Apply contrast adjustment
        adjusted = cv2.convertScaleAbs(clahe_img, alpha=contrast, beta=-25)

        # Ensure the result is properly clipped to 0-255 range
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        # Enhance shadows and highlights
        shadow = cv2.multiply(adjusted, adjusted, scale=1/255.0)
        highlight = cv2.multiply(255-adjusted, 255-adjusted, scale=1/255.0)
        enhanced = cv2.addWeighted(adjusted, 0.6, shadow, 0.2, 0)
        enhanced = cv2.addWeighted(enhanced, 0.8, 255-highlight, 0.2, 0)

        # Convert back to 3-channel image (but still grayscale)
        return cv2.cvtColor(adjusted, cv2.COLOR_GRAY2BGR)
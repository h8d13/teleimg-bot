import cv2
import numpy as np
from .base_model import BaseModel

class EnhanceDarkModel(BaseModel):
    @staticmethod
    def process_image(image, clip_limit=2.0, tile_grid_size=(8,8), saturation_boost=10, sharpen_strength=1.5):
        # Convert to LAB color space for better brightness and contrast control
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to the L channel (adaptive brightness/contrast adjustment)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)

        # Normalize the L channel using adaptive gamma correction
        cl = cv2.normalize(cl, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # Adaptive saturation adjustment (A and B channels)
        a = cv2.add(a, saturation_boost)
        b = cv2.add(b, saturation_boost)

        # Merge LAB channels back together
        merged_lab = cv2.merge((cl, a, b))
        enhanced_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

        # Optional noise reduction before sharpening
        enhanced_image = cv2.fastNlMeansDenoisingColored(enhanced_image, None, 10, 10, 7, 21)

        # Advanced sharpening using unsharp masking
        gaussian = cv2.GaussianBlur(enhanced_image, (0, 0), sigmaX=2)
        sharpened_image = cv2.addWeighted(enhanced_image, 1 + sharpen_strength, gaussian, -sharpen_strength, 0)

        return sharpened_image


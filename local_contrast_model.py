import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_model import BaseModel

class LocalContrastEnhancementModel(BaseModel):
    @staticmethod
    def _enhance_local_contrast(image, kernel_size=8, clip_limit=1.5):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(kernel_size, kernel_size))
        cl = clahe.apply(l)
        
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced

    @staticmethod
    def _subtle_sharpen(image, amount=0.3):
        blurred = cv2.GaussianBlur(image, (0, 0), 1)
        sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
        return sharpened

    @staticmethod
    def _blend_with_original(original, enhanced, alpha=0.3):
        return cv2.addWeighted(original, 1 - alpha, enhanced, alpha, 0)

    @staticmethod
    def calculate_enhancement_score(original, enhanced):
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        original_std = np.std(original_gray)
        enhanced_std = np.std(enhanced_gray)
        contrast_improvement = enhanced_std / original_std if original_std > 0 else 1
        
        original_sharpness = cv2.Laplacian(original_gray, cv2.CV_64F).var()
        enhanced_sharpness = cv2.Laplacian(enhanced_gray, cv2.CV_64F).var()
        sharpness_improvement = enhanced_sharpness / original_sharpness if original_sharpness > 0 else 1
        
        return 0.5 * contrast_improvement + 0.5 * sharpness_improvement

    @staticmethod
    def auto_enhance(image, kernel_range=(5, 15, 2), clip_limit_range=(1.0, 2.0, 0.2), amount_range=(0.1, 0.4, 0.1), alpha_range=(0.1, 0.5, 0.1)):
        best_score = -float('inf')
        best_params = None
        best_enhanced = None

        for kernel_size in range(*kernel_range):
            for clip_limit in np.arange(*clip_limit_range):
                for amount in np.arange(*amount_range):
                    for alpha in np.arange(*alpha_range):
                        enhanced = LocalContrastEnhancementModel.process_image(image, kernel_size, clip_limit, amount, alpha)
                        score = LocalContrastEnhancementModel.calculate_enhancement_score(image, enhanced)
                        
                        if score > best_score:
                            best_score = score
                            best_params = (kernel_size, clip_limit, amount, alpha)
                            best_enhanced = enhanced

        return best_enhanced, best_params

    @staticmethod
    def process_image(image, kernel_size=8, clip_limit=1.5, amount=0.5, alpha=0.66, auto_enhance=False):
        if auto_enhance:
            return LocalContrastEnhancementModel.auto_enhance(image)
        
        enhanced = LocalContrastEnhancementModel._enhance_local_contrast(image, kernel_size, clip_limit)
        sharpened = LocalContrastEnhancementModel._subtle_sharpen(enhanced, amount)
        result = LocalContrastEnhancementModel._blend_with_original(image, sharpened, alpha)
        
        return result
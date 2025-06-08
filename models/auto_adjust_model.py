import cv2
import numpy as np
from scipy import interpolate
from .base_model import BaseModel

class AutoAdjustModel(BaseModel):
    @staticmethod
    def analyze_histogram(channel):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        cdf = hist.cumsum()
        return hist, cdf

    @staticmethod
    def find_significant_range(cdf, low_percentile=0.02, high_percentile=0.98):
        low = np.argmax(cdf > low_percentile)
        high = np.argmax(cdf > high_percentile)
        
        # If high is 0, it means cdf never reached high_percentile
        if high == 0:
            high = 255
        
        # Ensure low and high are different
        if low == high:
            if low > 0:
                low -= 1
            else:
                high += 1
        
        return low, high

    @staticmethod
    def create_adjustment_curve(low, high, strength=0.8):
        # Ensure low and high are different and in the correct order
        low = max(0, min(low, 254))  # Ensure low is between 0 and 254
        high = max(low + 1, min(high, 255))  # Ensure high is greater than low and at most 255
        
        x = [0, low, (low + high) / 2, high, 255]
        y = [0, 0, 128, 255, 255]
        
        # Ensure x values are unique and strictly increasing
        x, indices = np.unique(x, return_index=True)
        y = np.array(y)[indices]
        
        spline = interpolate.PchipInterpolator(x, y)
        curve = spline(np.arange(256))
        identity = np.arange(256)
        return curve * strength + identity * (1 - strength)

    @staticmethod
    def adjust_channel(channel, curve):
        return cv2.LUT(channel, curve.astype(np.uint8))

    @staticmethod
    def process_image(image):
        adjusted = np.zeros_like(image)
        for i in range(3):  # For each channel
            hist, cdf = AutoAdjustModel.analyze_histogram(image[:,:,i])
            low, high = AutoAdjustModel.find_significant_range(cdf)
            curve = AutoAdjustModel.create_adjustment_curve(low, high)
            adjusted[:,:,i] = AutoAdjustModel.adjust_channel(image[:,:,i], curve)
        return adjusted
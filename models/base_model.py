# /models base_model.py
import time

class BaseModel:
    @staticmethod
    def process_image(image):
        time.sleep(5)
        raise NotImplementedError("This method should be implemented by subclasses")
## /models/__init__.py

## LOAD OTHER MODELS THROUGH 
## SIMULATE 5 SECOND PROCESSING TIME FOR EACH MODEL IN BASE_MODEL.PY

from .auto_adjust_model import AutoAdjustModel
from .simple_contrast_model import SimpleContrastModel
from .black_and_white_model import BlackAndWhiteModel
from .upscale_model import UpscaleModel
from .downscale_model import DownscaleModel
from .enhance_dark_model import EnhanceDarkModel
from .local_contrast_model import LocalContrastEnhancementModel

MODELS = {
    "auto": AutoAdjustModel,
    "contrast": SimpleContrastModel,
    "bw": BlackAndWhiteModel,
    "upscale": UpscaleModel,
    "downscale" : DownscaleModel,
    "darker": EnhanceDarkModel,
    "local contrast": LocalContrastEnhancementModel,
}
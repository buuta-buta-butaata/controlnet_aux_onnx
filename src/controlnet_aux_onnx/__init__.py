__version__ = "0.1.0"

from .detector import Detector

from .leres import LeresDetector
from .midas import MidasDetector

from .hed import HEDdetector
from .pidi import PidiNetDetector
from .teed import TEEDdetector
from .zoe_with_filpped import ZoeDetector
# from .zoe import ZoeDetector


__all__ = [
    # "Detector",
    "LineartDetector",
    "LineartCoarseDetector",
    "LineartAnimeDetector",
    "MidasDetector",
    # "MLSDdetector",
    "NormalBaeDetector",

    "LeresDetector",
    "ZoeDetector",

    "HEDdetector",
    "PidiNetDetector",
    "TEEDdetector",
]

class LineartDetector(Detector):
    def __init__(self, model_dir="./models", providers=None, provider_options=None, sess_options=None):
        super().__init__("lineart",
                         model_dir=model_dir,
                         providers=providers,
                         provider_options=provider_options,
                         sess_options=sess_options)

class LineartCoarseDetector(Detector):
    def __init__(self, model_dir="./models", providers=None, provider_options=None, sess_options=None):
        super().__init__("lineart_coarse",
                         model_dir=model_dir,
                         providers=providers,
                         provider_options=provider_options,
                         sess_options=sess_options)

class LineartAnimeDetector(Detector):
    def __init__(self, model_dir="./models", providers=None, provider_options=None, sess_options=None):
        super().__init__("lineart_anime",
                         model_dir=model_dir,
                         providers=providers,
                         provider_options=provider_options,
                         sess_options=sess_options)
        self.divisible_factor = 256

# 他のモデルとは異なり、torchの処理で画像ではなく線分の座標を出力してくるので、後回し
# class MLSDdetector(Detector):
#     def __init__(self, model_dir="./models", providers=None, provider_options=None, sess_options=None):
#         super().__init__("mlsd",
#                          model_dir=model_dir,
#                          providers=providers,
#                          provider_options=provider_options,
#                          sess_options=sess_options)

class NormalBaeDetector(Detector):
    def __init__(self, model_dir="./models", providers=None, provider_options=None, sess_options=None):
        super().__init__("normalbae",
                         model_dir=model_dir,
                         providers=providers,
                         provider_options=provider_options,
                         sess_options=sess_options)

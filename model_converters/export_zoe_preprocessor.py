import torch
from torch.export import Dim
import torch.nn as nn
import torch.nn.functional as F

from util import export_onnx


def export():
    name = "zoe_preprocessor"
    shape = (2, 512, 512, 3)

    batch = Dim("batch", min=1, max=2)
    _height = Dim('_height', min=1, max=16)
    _width = Dim('_width', min=1, max=16)
    height = 32 * _height
    width = 32 * _width
    dynamic_shapes = {"image": {0: batch, 1: height, 2: width}}
    output_names = ["normalized_image", "normalized_image_flipped"]

    wrapper = Wrapper()
    wrapper.eval()

    export_onnx(name, wrapper, shape,
                dynamic_shapes=dynamic_shapes,
                output_names=output_names)

class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def normalize(self, tensor):
        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
        tensor = (tensor - mean) / std
        return tensor

    def forward(self, image):
        N, H, W, _ = image.shape

        image = torch.flip(image, dims=[-1])
        image = image.to(torch.float32)
        image = image / 255.0
        image = self.normalize(image).permute(0, 3, 1, 2)

        # padding
        pad_height = int((H // 2) ** 0.5 * 3)
        pad_width = int((W // 2) ** 0.5 * 3)

        image = F.pad(image, (pad_width, pad_width, pad_height, pad_height), mode="reflect")
        image = F.interpolate(image, size=(H, W), mode="bilinear", align_corners=True)

        image_flipped = torch.flip(image, dims=[3])

        return image, image_flipped

if __name__ == "__main__":
    export()

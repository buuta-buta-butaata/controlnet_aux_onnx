import torch
from torch.export import Dim
import torch.nn as nn

from util import export_onnx


def export():
    name = "zoe_postprocessor"
    shape = (2, 1, 512, 512)

    batch = Dim("batch", min=1, max=2)
    _height = Dim('_height', min=1, max=16)
    _width = Dim('_width', min=1, max=16)
    height = 32 * _height
    width = 32 * _width

    dummy_input = (
        torch.randn(shape, dtype=torch.float32),
        torch.randn(shape, dtype=torch.float32),
    )
    dynamic_shapes={
        "depth": {0: batch, 2: height, 3: width},
        "depth_flipped": {0: batch, 2: height, 3: width},
    }
    input_names = ["depth", "depth_flipped"]

    wrapper = Wrapper()
    export_onnx(name, wrapper, shape,
                dynamic_shapes=dynamic_shapes,
                dummy_input=dummy_input,
                input_names=input_names)

class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, depth, depth_flipped):
        depth_flipped = torch.flip(depth_flipped, dims=[3])

        depth = (depth + depth_flipped) / 2.0

        vmin = torch.quantile(depth, 0.02)
        vmax = torch.quantile(depth, 0.85)
        depth = torch.where(vmax - vmin > torch.finfo(torch.float32).eps,
                          (depth - vmin) / (vmax - vmin),
                          torch.zeros(depth.shape))

        depth = 1.0 - depth
        depth = torch.clip(depth, 0.0, 1.0)

        # if gamma_corrected:
        #     depth = np.power(depth, 2.2)
        # depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)

        # N, 1, H, W -> N, 3, H, W
        output = depth.repeat(1, 3, 1, 1)

        return output

if __name__ == "__main__":
    export()

import torch
from torch.export import Dim

from util import export_onnx, batch, height, width


def export():
    name = "helper_rgb_float32_to_bgr_uint8"
    shape = (1, 3, 512, 512)

    dummy_input = torch.randn(shape, dtype=torch.float32)
    dynamic_shapes = {"normalized_image": {0: batch, 2: height, 3: width}}
    input_names = ["normalized_image"]

    wrapper = Wrapper()

    export_onnx(name, wrapper, shape,
                dummy_input=dummy_input,
                dynamic_shapes=dynamic_shapes,
                input_names=input_names)

class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, normalized_image):
        """ normalized_image: (dtype=float32 [0.0-1.0]) NCHW """
        bgr_image = torch.flip(normalized_image, dims=[1]) # RGB -> BGR
        output = (bgr_image * 255.0).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1) # NCHW -> NHWC
        return output

if __name__ == "__main__":
    export()
    # result = wrapper.forward(*dummy_input)
    # print(result.shape)
    # print(result)

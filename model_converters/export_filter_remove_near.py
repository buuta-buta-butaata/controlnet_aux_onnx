import torch
from torch.export import Dim

from util import export_onnx, batch, height, width


def export():
    name = "filter_remove_near"
    shape = (1, 3, 512, 512)
    threshold = 1

    dynamic_shapes = {
        "normalized_image": {0: batch, 2: height, 3: width},
        "removal_threshold": {0: threshold},
    }
    dummy_input = (
        torch.randn(shape, dtype=torch.float32),
        torch.randint(low=0, high=101, size=(1, )),
    )
    input_names = ["normalized_image", "removal_threshold"]

    wrapper = Wrapper()

    export_onnx(name, wrapper, shape,
                dummy_input=dummy_input,
                dynamic_shapes=dynamic_shapes,
                input_names=input_names)

class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, normalized_image, removal_threshold):
        """
        normalized_image: (dtype=float32 [0.0-1.0]) NCHW
        removal_threshold: dtype=int64 [0-100]
        """
        removal_threshold = 1.0 - (removal_threshold / 100.0)
        normalized_image[removal_threshold < normalized_image] = 1.0
        return normalized_image

if __name__ == "__main__":
    export()
    # result = wrapper.forward(*dummy_input)
    # print(result.shape)
    # print(result)

import torch
from torch.export import Dim

from util import export_onnx, batch, height, width


def export():
    name = "filter_safe_step"
    shape = (1, 3, 512, 512)
    mode = 1

    dynamic_shapes = {
        "normalized_image": {0: batch, 2: height, 3: width},
        "step": {0: mode},
    }
    dummy_input = (
        torch.randn(shape, dtype=torch.float32),
        torch.randint(low=0, high=2, size=(1, )),
    )
    input_names = ["normalized_image", "step"]

    wrapper = Wrapper()

    export_onnx(name, wrapper, shape,
                dummy_input=dummy_input,
                dynamic_shapes=dynamic_shapes,
                input_names=input_names)

class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # From: https://github.com/huggingface/controlnet_aux/blob/v0.0.10/src/controlnet_aux/util.py#L64-L67
    def safe_step(self, x, step):
        y = x * float(step + 1)
        y = y.to(torch.int32).to(torch.float32) / float(step)
        return y

    def forward(self, normalized_image, step):
        """ normalized_image: (dtype=float32 [0.0-1.0]) NCHW """
        output = self.safe_step(normalized_image, step).clip(0.0, 1.0)
        return output

if __name__ == "__main__":
    export()
    # result = wrapper.forward(*dummy_input)
    # print(result.shape)
    # print(result)


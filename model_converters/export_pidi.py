import torch
from torch.export import Dim
from controlnet_aux.pidi import PidiNetDetector

from util import export_onnx


def export():
    name = "pidi"
    model = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
    shape = (1, 512, 512, 3)

    wrapper = Wrapper(model)
    export_onnx(name, wrapper, shape)

class Wrapper(torch.nn.Module):
    def __init__(self, orig_model):
        super().__init__()
        self.orig_model = orig_model

    def forward(self, image):
        # Input: BGR
        # NHWC(uint8)[0-255] -> NCHW(float32)[0.0-1.0]
        image = image.to(torch.float32)
        image = image / 255.0
        image = image.permute(0, 3, 1, 2)
        edge = self.orig_model.netNetwork(image)[-1]

        # 1, 1, H, W -> 1, 3, H, W
        output = edge.repeat(1, 3, 1, 1)

        return output

if __name__ == "__main__":
    export()
    # dummy_input = torch.randint(low=0, high=256, size=shape, dtype=torch.uint8)
    # result = wrapper.forward(dummy_input)
    # print(result.shape)
    # print(result)


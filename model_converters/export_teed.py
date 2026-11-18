import torch
from torch.export import Dim
from torch.nn import functional as F
from controlnet_aux.teed import TEEDdetector

from util import export_onnx


def export():
    name = "teed"
    try:
        model = TEEDdetector.from_pretrained("../models", filename="7_model.pth")
    except FileNotFoundError:
        print("\nERROR: Model file '7_model.pth' not found.\n"
              "Please download from: https://github.com/xavysp/TEED/tree/main/checkpoints/BIPED/7\n"
              "To: ../models")
        exit()
    shape = (1, 512, 512, 3)

    wrapper = Wrapper(model)

    export_onnx(name, wrapper, shape)

class Wrapper(torch.nn.Module):
    def __init__(self, orig_model):
        super().__init__()
        self.orig_model = orig_model

    def forward(self, image):
        # NHWC(BGR, uint8) -> NCHW(RGB, float32)
        N, H, W, _ = image.shape
        image = torch.flip(image, dims=[-1])
        image = image.to(torch.float32)
        image = image.permute(0, 3, 1, 2)

        edges = self.orig_model.model(image)

        edges = [F.interpolate(e, (H, W), mode="bilinear") for e in edges]
        edges = torch.stack(edges, axis=2)
        edge = torch.sigmoid(torch.mean(edges, axis=2))
        # 1, 1, H, W -> 1, 3, H, W
        output = edge.repeat(1, 3, 1, 1)

        return output

if __name__ == "__main__":
    export()
    # dummy_input = torch.randint(low=0, high=256, size=shape, dtype=torch.uint8)
    # result = wrapper.forward(dummy_input)
    # print(result.shape)
    # print(result)

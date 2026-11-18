import torch
from torch.export import Dim
from controlnet_aux.normalbae import NormalBaeDetector

from util import export_onnx


def export():
    name = "normalbae"
    model = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
    # batch=1だと1で固定されてエクスポートされてしまうため、batch=2を指定
    shape = (2, 512, 512, 3)

    wrapper = Wrapper(model)
    export_onnx(name, wrapper, shape)

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def normalize(self, tensor):
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        tensor = (tensor - mean) / std
        return tensor

    def forward(self, image):
        # NHWC(BGR, uint8)[0-255] -> NCHW(RGB, float32)[0.0-1.0]
        image = torch.flip(image, dims=[-1])
        image = image.to(torch.float32) / 255.0
        image = self.normalize(image).permute(0, 3, 1, 2)

        normal = self.model.model(image)
        normal = normal[0][-1][:, :3]
        normal = ((normal + 1) * 0.5).clip(0, 1)

        return normal

if __name__ == "__main__":
    export()
    # dummy_input = torch.randint(low=0, high=256, size=shape, dtype=torch.uint8)
    # result = wrapper.forward(dummy_input)
    # print(result.shape)
    # print(result)

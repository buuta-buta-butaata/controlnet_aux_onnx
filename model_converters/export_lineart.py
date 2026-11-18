import torch
from torch.export import Dim
from controlnet_aux.lineart import LineartDetector

from util import export_onnx


def export():
    name = "lineart"
    model = LineartDetector.from_pretrained("lllyasviel/Annotators")
    shape = (1, 512, 512, 3)

    wrapper = Wrapper(model)

    export_onnx(name, wrapper, shape)

class Wrapper(torch.nn.Module):
    def __init__(self, orig_model):
        super().__init__()
        self.orig_model = orig_model

    def forward(self, image):
        # NHWC(BGR, uint8)[0-255] -> NCHW(RGB, float32)[0.0-1.0]
        image = torch.flip(image, dims=[-1])
        image = image.to(torch.float32)
        image = image / 255.0
        image = image.permute(0, 3, 1, 2)
        line = self.orig_model.model(image)

        # 色を反転させる
        # controlnet_auxの出力とできる限り一致させるため、いったんuint8にする
        line = (line * 255.0).clip(0, 255).to(torch.uint8)
        line = torch.bitwise_not(line)
        line = line.to(torch.float32) / 255.0

        # 1, 1, H, W -> 1, 3, H, W
        output = line.repeat(1, 3, 1, 1)

        return output

if __name__ == "__main__":
    export()
    # dummy_input = torch.randint(low=0, high=256, size=shape, dtype=torch.uint8)
    # result = wrapper.forward(dummy_input)
    # print(result.shape)
    # print(result)

import torch
from torch.export import Dim
from torch.nn import functional as F
from controlnet_aux.hed import ControlNetHED_Apache2, HEDdetector

from util import export_onnx


def export():
    name = "hed"
    model = HEDdetector.from_pretrained("lllyasviel/Annotators")
    shape = (1, 512, 512, 3)

    wrapper = Wrapper(model)
    export_onnx(name, wrapper, shape)

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        N, H, W, _ = image.shape
        # QNN HTPだと、uint8->float32の変換でおかしなふるまいをする
        # n=0 -> 0.0、n>1 -> 1.0みたいな変換をしてしまう
        # いったんuint8 -> uint32にすると、正常に動く
        # image = image.to(torch.uint32)
        # HEDは / 255.0は不要
        # NHWC(BGR, uint8) -> NCHW(RGB, float32)
        image = torch.flip(image, dims=[-1])
        image = image.to(torch.float32).permute(0, 3, 1, 2)
        edges = self.model.netNetwork(image)
        edges = [F.interpolate(e, (H, W), mode="bilinear") for e in edges]
        edges = torch.stack(edges, axis=2)
        edge = torch.sigmoid(torch.mean(edges, axis=2))
        output = edge.repeat(1, 3, 1, 1)
        return output

if __name__ == "__main__":
    export()
    # from PIL import Image
    # import numpy as np
    # img = Image.open(r"pose512.png").convert("RGB")
    # img_np = np.array(img)
    # input_data = img_np[np.newaxis, :, :, :]

    # print(input_data.shape)

    # out = wrapper(torch.from_numpy(input_data)).detach().numpy()[0]
    # im = Image.fromarray((out.transpose(1,2,0) * 255.0).clip(0, 255).astype(np.uint8))
    # im.save("test.png")

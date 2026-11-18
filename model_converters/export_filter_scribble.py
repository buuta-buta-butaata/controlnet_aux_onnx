import numpy as np
import torch
from torch.export import Dim

import kornia.morphology as morph
import kornia.filters as filters

from util import export_onnx, batch, height, width


def export():
    name = "filter_scribble"
    shape = (2, 3, 512, 512)

    dummy_input = torch.randn(shape, dtype=torch.float32)
    dynamic_shapes={"normalized_image": {0: batch, 2: height, 3: width}}
    input_names = ["normalized_image"]

    wrapper = Wrapper()

    export_onnx(name, wrapper, shape,
                dummy_input=dummy_input,
                dynamic_shapes=dynamic_shapes,
                input_names=input_names)

class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def gaussian_blur(self, image, sigma, dtype=torch.float32):
        image = filters.gaussian_blur2d(image.to(torch.float32), 25, (3.0, 3.0), separable=False).to(dtype)

        return image

    # From: https://github.com/huggingface/controlnet_aux/blob/v0.0.10/src/controlnet_aux/pidi/__init__.py#L76-L79
    def scribble(self, edge):
        edge = self.nms(edge, 127, 3.0)
        edge = self.gaussian_blur(edge, 3.0, torch.uint8)
        edge[edge > 4] = 255
        edge[edge < 255] = 0
        return edge

    # From: https://github.com/huggingface/controlnet_aux/blob/v0.0.10/src/controlnet_aux/util.py#L41-L56
    def nms(self, x, t, s):
        x = self.gaussian_blur(x, s)

        f1 = torch.tensor([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=torch.uint8)
        f2 = torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=torch.uint8)
        f3 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.uint8)
        f4 = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.uint8)

        y = torch.zeros_like(x)

        for f in [f1, f2, f3, f4]:
            mask = morph.dilation(x, kernel=f, border_type="constant") == x
            y = torch.where(mask, x, y)

        z = torch.zeros_like(y, dtype=torch.uint8)
        z[y > t] = 255
        return z

    def forward(self, normalized_image):
        """ normalized_image: (dtype=float32 [0.0-1.0]) NCHW """
        normalized_image = (normalized_image * 255.0).clip(0, 255).to(torch.uint8)
        normalized_image = normalized_image[:, :1, :, :]
        edge = self.scribble(normalized_image).repeat(1, 3, 1, 1)
        output = edge.to(torch.float32) / 255.0
        return output

if __name__ == "__main__":
    export()

    # from PIL import Image
    # image = Image.open(r"Z:\Work\lab\input_scribble.png")
    # image = np.array(image.convert("RGB"))
    # image_tensor = torch.from_numpy(np.array([(image.astype(np.float32) / 255.0).transpose(2, 0, 1)]).astype(np.float32))
    # print(image_tensor.shape)
    # output = wrapper.forward(image_tensor)[0]
    # print(output.shape)
    # Image.fromarray((output.detach().numpy() * 255.0).clip(0, 255).transpose(1, 2, 0).astype(np.uint8)).save("scribble.png")


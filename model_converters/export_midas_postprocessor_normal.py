import torch
import torch.nn.functional as F
from torch.export import Dim

from util import export_onnx


def export():
    name = "midas_postprocessor_normal"
    # batch=1だと1で固定されてエクスポートされてしまうため、batch=2を指定
    shape = (2, 512, 512)

    batch = Dim("batch", min=1, max=2)
    _height = Dim('_height', min=1, max=16)
    _width = Dim('_width', min=1, max=16)
    height = 32 * _height
    width = 32 * _width
    norm = 1
    thr = 1
    dynamic_shapes = {
        "raw_depth": {0: batch, 1: height, 2: width},
        "a": {0: norm},
        "bg_th": {0: thr},
    }
    dummy_input = (
        torch.randn(shape, dtype=torch.float32),
        torch.randn((1, ), dtype=torch.float32),
        torch.randn((1, ), dtype=torch.float32),
    )
    input_names = ["raw_depth", "a", "bg_th"]

    wrapper = Wrapper()
    # wrapper.forward(*dummy_input)

    export_onnx(name, wrapper, shape,
                dummy_input=dummy_input,
                dynamic_shapes=dynamic_shapes,
                input_names=input_names)


class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def sobel(self, image, kernel):
        N, _, H, W = image.shape
        sobel_filter = kernel.expand(1, 1, 3, 3)
        image = F.pad(image, (1, 1, 1, 1), mode="reflect")
        image = F.conv2d(image, sobel_filter)
        return image

    def forward(self, raw_depth, a, bg_th):
        """
        raw_depth: (dtype=float32 [0.0-1.0]) NHW
        """
        v_kernel = torch.FloatTensor([[-1, -2, -1],
                                      [0, 0, 0],
                                      [1, 2, 1]])
        h_kernel = torch.FloatTensor([[-1, 0, 1],
                                      [-2, 0, 2],
                                      [-1, 0, 1]])

        depth = torch.unsqueeze(raw_depth, 1)
        depth_pt = depth.clone()
        depth_pt -= torch.min(depth_pt)
        depth_pt /= torch.max(depth_pt)

        x = self.sobel(depth, h_kernel)
        y = self.sobel(depth, v_kernel)
        x[depth_pt < bg_th] = 0
        y[depth_pt < bg_th] = 0
        x = x.squeeze(1)
        y = y.squeeze(1)
        z = torch.ones_like(x) * a
        normal = torch.stack([x, y, z], dim=1)
        normal /= torch.sum(normal ** 2.0, dim=1, keepdims=True) ** 0.5
        normal = normal * 0.5 + 0.5
        output = torch.flip(normal, dims=[1])

        return output

if __name__ == "__main__":
    export()
    # from PIL import Image
    # import numpy as np
    # path = r"Z:\Work\lab\controlnet_aux_onnx\tests\test_images\ninja512.png"
    # # path = r"pose512.png"
    # img = Image.open(path).convert("RGB")
    # img = img.resize((512, 512))
    # img_np = np.array(img)
    # input_data = img_np[np.newaxis, :, :, :]

    # print(input_data.shape)

    # wrapper = Wrapper()
    # out = wrapper.forward(torch.from_numpy(input_data), a=np.pi * 2.0, bg_th=0.1).detach().numpy()

    # print(out[0].shape)

    # im = Image.fromarray((out[0].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8))
    # im.save("test.png")

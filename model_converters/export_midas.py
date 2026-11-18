import torch
from torch.export import Dim
from controlnet_aux.midas import MidasDetector
from controlnet_aux.midas.api import MiDaSInference

from util import export_onnx


def export():
    name = "midas"
    model = MidasDetector.from_pretrained("lllyasviel/Annotators")
    # batch=1だと1で固定されてエクスポートされてしまうため、batch=2を指定
    shape = (2, 512, 512, 3)

    batch = Dim("batch", min=1, max=2)
    _height = Dim('_height', min=1, max=16)
    _width = Dim('_width', min=1, max=16)
    height = 32 * _height
    width = 32 * _width
    dynamic_shapes = {"image": {0: batch, 1: height, 2: width}}
    output_names = ["output", "raw_depth"]

    wrapper = Wrapper(model)

    export_onnx(name, wrapper, shape, dynamic_shapes=dynamic_shapes, output_names=output_names)


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        # NHWC(BGR, uint8)[0-255] -> NCHW(RGB, float32)[-1.0-1.0]
        image = torch.flip(image, dims=[-1])
        image = image.to(torch.float32)
        image = image / 127.5 - 1.0
        image = image.permute(0, 3, 1, 2)

        with torch.no_grad():
            depth = self.model.model(image)
        depth_pt = depth.clone()
        depth_pt -= torch.min(depth_pt)
        depth_pt /= torch.max(depth_pt)
        output = torch.unsqueeze(depth_pt, 1)
        output = output.repeat(1, 3, 1, 1)

        return output, depth

if __name__ == "__main__":
    export()
    # from PIL import Image
    # import numpy as np
    # path = r"Z:\Work\lab\controlnet_aux_onnx\tests\test_images\flowery_room.png"
    # # path = r"pose512.png"
    # img = Image.open(path).convert("RGB")
    # img = img.resize((512, 512))
    # img_np = np.array(img)
    # input_data = img_np[np.newaxis, :, :, :]

    # print(input_data.shape)

    # out = wrapper(torch.from_numpy(input_data)).detach().numpy()

    # print(out[0].shape)

    # im = Image.fromarray((out[0].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8))
    # im.save("test.png")

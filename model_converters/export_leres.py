import torch
from torch.export import Dim
from controlnet_aux.leres import LeresDetector

from util import export_onnx


def export():
    name = "leres"
    model = LeresDetector.from_pretrained("lllyasviel/Annotators")
    # batch=1だと1で固定されてエクスポートされてしまうため、batch=2を指定
    shape = (2, 512, 512, 3)

    wrapper = Wrapper(model)
    export_onnx(name, wrapper, shape)

def freeze_batchnorm_for_leres(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            # 💡 evalモードにする代わりに「常に今入力された画像の統計値を使う」ように強制する
            m.train()                     # 内部の挙動をトレーニング状態（リアルタイム計算）にする
            m.track_running_stats = False # 過去の蓄積データを更新・参照させない
            # 重み（weight, bias）が勝手に更新されないように勾配計算を切る
            if m.weight is not None: m.weight.requires_grad = False
            if m.bias is not None: m.bias.requires_grad = False

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
        # NHWC(BGR, uint8) -> NCHW(BGR, float32)
        image = image.to(torch.float32)
        image = self.normalize(image).permute(0, 3, 1, 2)

        freeze_batchnorm_for_leres(self.model.model.depth_model)
        with torch.no_grad():
            depth = self.model.model.depth_model(image)

        depth_min = depth.min()
        depth_max = depth.max()
        max_val = torch.iinfo(torch.uint16).max

        out = torch.where(depth_max - depth_min > torch.finfo(torch.float32).eps,
                          max_val * (depth - depth_min) / (depth_max - depth_min),
                          torch.zeros(depth.shape))

        depth_image = out.clip(0, max_val).to(torch.uint16).to(torch.float32)

        # convert to uint8
        depth_image = torch.round(depth_image * 255.0 / 65535.0).clip(0, 255).to(torch.uint8)
        depth_image = torch.bitwise_not(depth_image)
        depth_image = depth_image.to(torch.float32) / 255.0
        output = depth_image.repeat(1, 3, 1, 1)

        return output

if __name__ == "__main__":
    export()
    # from PIL import Image
    # import numpy as np
    # path = r"ninja512.png"
    # path = r"test_input.png"
    # img = Image.open(path).convert("RGB").resize((512, 512), Image.Resampling.BOX)
    # img_np = np.array(img)
    # input_data = img_np[np.newaxis, :, :, :]
    # input_data = input_data[..., ::-1].copy()
    # print(input_data.shape)

    # out = wrapper(torch.from_numpy(input_data)).detach().numpy()[0]
    # im = Image.fromarray((out.transpose(1,2,0) * 255.0).clip(0, 255).astype(np.uint8))
    # im.save("test.png")

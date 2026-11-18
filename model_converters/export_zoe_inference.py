import torch
from torch.export import Dim
import torch.nn as nn
from torchvision.transforms.v2 import functional as tvF
from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor

from util import export_onnx


def export():
    BeitAttention.forward = forward_contiguous

    name = "zoe_inference"
    model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu")
    shape = (2, 3, 512, 512)

    batch = Dim("batch", min=1, max=2)
    _height = Dim('_height', min=1, max=16)
    _width = Dim('_width', min=1, max=16)
    height = 32 * _height
    width = 32 * _width

    dummy_input = (
        torch.randn(shape, dtype=torch.float32),
    )
    dynamic_shapes={
        "normalized_image": {0: batch, 2: height, 3: width},
    }
    input_names = ["normalized_image"]
    output_names = ["depth"]

    wrapper = Wrapper(model)
    # デフォルトのoptimize=Trueだと、batchがdynamic shapeにならない
    # そのため、optimize=Falseでエクスポートし、onnxslimを後で実行する
    # export_onnx(name, wrapper, shape, dynamic_shapes=dynamic_shapes, optimize=False)
    export_onnx(name, wrapper, shape,
                dynamic_shapes=dynamic_shapes,
                dummy_input=dummy_input,
                input_names=input_names,
                output_names=output_names,
                # optimize=False,
                # slim=False,
                output_dir=r"C:\Applications\AI\github\controlnet_aux_onnx\models")
    # export_onnx(name, wrapper, shape, output_dir=r"C:\Applications\AI\github\controlnet_aux_onnx\models", dynamic_shapes=None)


# for transformers Version: 5.22.2
# contiguous()がないとエクスポートに失敗するので、追加しておく
from transformers.models.beit.modeling_beit import BeitAttention, eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

def forward_contiguous(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2).contiguous()
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2).contiguous()
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2).contiguous()

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def infer(self, image, H, W, pad_height, pad_width):
        depth = self.model(pixel_values=image).predicted_depth.contiguous()
        # N, H, W -> N, 1, H, W
        depth = torch.unsqueeze(depth, 1)

        # remove padding
        depth = tvF.resize(
            depth,
            size=[H + 2 * pad_height, W + 2 * pad_width],
            interpolation=tvF.InterpolationMode.BICUBIC,
            antialias=False,
        )

        # エクスポートに失敗するので、分割する
        # depth = depth[:, :, pad_height:-pad_height, pad_width:-pad_width]
        depth = depth[:, :, pad_height:-pad_height, :]
        depth = depth[:, :, :, pad_width:-pad_width]

        return depth

    def forward(self, normalized_image):
        N, _, H, W = normalized_image.shape

        # padding
        pad_height = int((H // 2) ** 0.5 * 3)
        pad_width = int((W // 2) ** 0.5 * 3)

        depth = self.infer(normalized_image, H, W, pad_height, pad_width)
        return depth

if __name__ == "__main__":
    export()
    # from PIL import Image
    # import numpy as np
    # path = r"pose512.png"
    # img = Image.open(path).convert("RGB")
    # img = img.resize((512, 512))
    # img_np = np.array(img)[..., ::-1]
    # input_data = img_np[np.newaxis, :, :, :].copy()

    # print(input_data.shape)

    # out = wrapper(torch.from_numpy(input_data)).detach().numpy()

    # print(out[0].shape)

    # im = Image.fromarray((out[0].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8))
    # im.save("test.png")

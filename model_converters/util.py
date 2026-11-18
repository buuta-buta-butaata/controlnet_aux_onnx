import os

import torch
from torch.export import Dim


batch = Dim("batch", min=1, max=2)
height = Dim("height", min=511, max=512)
width = Dim("width", min=511, max=512)

def export_onnx(
        model_name,
        wrapper,
        shape=(1, 1024, 1024, 3),
        dynamic_shapes={"image": {0: batch, 1: height, 2: width}},
        output_dir=f"{os.path.dirname(__file__)}/../models",
        dummy_input=None,
        input_names=["image"],
        output_names=["output"],
        opset=22,
        optimize=True,
        report=False,
        slim=True,
        **kwargs):
    wrapper.eval()

    os.makedirs(output_dir, exist_ok=True)
    onnx_path = f"{output_dir}/{model_name}.onnx"
    if dummy_input is None:
        dummy_input = torch.randint(low=0, high=256, size=shape, dtype=torch.uint8)

    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        external_data=False,
        input_names=input_names,
        dynamic_shapes=dynamic_shapes,
        output_names=output_names,
        optimize=optimize,
        report=report,
    )

    if slim:
        import onnx
        import onnxslim
        model = onnx.load(onnx_path)
        slimmed_model = onnxslim.slim(model, no_shape_infer=True)
        if slimmed_model:
            onnx.save(slimmed_model, onnx_path)

    print(f"{model_name} を {onnx_path} にエクスポートしました。")



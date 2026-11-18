import onnx
from onnx import TensorProto
import onnx_graphsurgeon as gs

import numpy as np


def fix_onnx_scale_types(onnx_path):
    model = onnx.load(onnx_path)
    graph = model.graph

    broken_graph = gs.import_onnx(model)
    for name, tensor in broken_graph.tensors().items():
        if "val_5549" in name:
            print(name)
            print(tensor)
            print(tensor.values)
            fixed_tensor = gs.Constant(
                name="val_5549",
                values=np.array(tensor.values, dtype=np.float32)
            )

    for node in broken_graph.nodes:
        if node.op == "Resize" and "val_5549" in node.inputs[2].name:
            node.inputs[2] = fixed_tensor
            print(node.inputs[2])

    new_model = gs.export_onnx(broken_graph)
    print(f"修正完了")
    return new_model

model_path = r"zoe_inference_slim.onnx"
output_path = r"zoe_inference.onnx"

fixed_model = fix_onnx_scale_types(model_path)

onnx.save(fixed_model, output_path)

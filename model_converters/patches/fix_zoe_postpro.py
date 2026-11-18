import onnx
from onnx import TensorProto
import onnx_graphsurgeon as gs


def fix_onnx_scale_types(onnx_path, output_path):
    model = onnx.load(onnx_path)
    graph = model.graph

    fixed_count = 0

    for node in graph.node:
        if node.op_type == "Cast" and node.output[0] == "sym_float":
            node.attribute[0].i = 11
            print(node.output)
            break

    broken_graph = gs.import_onnx(model)
    for name, tensor in broken_graph.tensors().items():
        if name == "sym_float":
            tensor.dtype = "float64"

    new_model = gs.export_onnx(broken_graph)
    onnx.save(new_model, output_path)
    print(f"修正完了")

model_path = r"zoe_postprocessor.onnx"

fix_onnx_scale_types(model_path, "zoe_postprocessor_fixed.onnx")

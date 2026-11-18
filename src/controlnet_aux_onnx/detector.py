import os
from PIL import Image

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from .util import resize_image, resize_to


class Detector:
    def __init__(self, model_name, model_dir="./models", providers=None, provider_options=None, sess_options=None):
        self.model_name = model_name
        self.divisible_factor = 64
        self.models = {}
        self.models[self.model_name] = self._load_model(model_name, model_dir, providers, provider_options, sess_options)

    def _load_model(self, model_name, model_dir, providers=None, provider_options=None, sess_options=None):
        onnx_path = os.path.join(model_dir, f"{model_name}.onnx")
        if not os.path.exists(onnx_path):
            print(onnx_path)
            hf_hub_download("Buuta/controlnet_aux_onnx", f"{model_name}.onnx", local_dir=model_dir)
            # raise FileNotFoundError(f"Onnx not found: {onnx_path}")

        session = ort.InferenceSession(
            path_or_bytes=onnx_path, providers=providers, provider_options=provider_options, sess_options=sess_options
        )
        input_names = list(map(lambda x: x.name, session.get_inputs()))
        output_names = list(map(lambda x: x.name, session.get_outputs()))

        return session, input_names, output_names

    def _infer(self, models, model_name, input_data, **kwargs):
        session, input_names, output_names = models[model_name]
        onnx_input = {input_names[0]: input_data}
        outputs = session.run(output_names, onnx_input)[0]
        return outputs

    def _filter(self, image, **kwargs):
        """
        Filter for onnx outputs
        :param image: NCHW float32 range[0, 1]
        :return: image NCHW float32 range[0, 1]
        """
        if isinstance(image, tuple):
            return (image[0].clip(0.0, 1.0), image[1].clip(0.0, 1.0))
        return image.clip(0.0, 1.0)

    def _filter_for_uint8(self, image, **kwargs):
        """
        Filter for onnx outputs
        :param image: HWC uint8 range[0, 255]
        :return: image HWC uint8 range[0, 255]
        """
        return image

    def __call__(
            self,
            input_image,
            detect_resolution=512,
            image_resolution=512,
            output_type="pil",
            **kwargs):
        if type(input_image) != np.ndarray:
            # RGB -> BGR
            input_image = np.array(input_image.convert("RGB"))[..., ::-1]

        original_width, original_height, _ = input_image.shape

        input_image = resize_image(input_image, detect_resolution, divisible_factor=self.divisible_factor)

        img_np = np.array(input_image)
        input_data = img_np[np.newaxis, :, :, :]

        outputs = self._infer(self.models, self.model_name, input_data, **kwargs)

        outputs = self._filter(outputs, **kwargs)

        sub_outputs = None
        if isinstance(outputs, tuple):
            output = self._postprocess(outputs[0][0], image_resolution, original_width, original_height, output_type, **kwargs)
            sub_output = self._postprocess(outputs[1][0], image_resolution, original_width, original_height, output_type, **kwargs)
            return output, sub_output

        return self._postprocess(outputs[0], image_resolution, original_width, original_height, output_type, **kwargs)

    def _postprocess(self, image, image_resolution, original_width, original_height, output_type, **kwargs):
        rgb = image.transpose(1, 2, 0) # (1, 3, H, W) -> (H, W, 3)
        rgb_uint8 = (rgb * 255.0).clip(0, 255).astype(np.uint8)

        rgb_uint8 = self._filter_for_uint8(rgb_uint8, **kwargs)

        if image_resolution > 0:
            rgb_uint8 = resize_image(rgb_uint8, image_resolution)
        else:
            # image_resolutionが設定されていない場合は、元の画像の解像度で出力する
            rgb_uint8 = resize_to(rgb_uint8, original_width, original_height)

        if output_type == "pil":
            return Image.fromarray(rgb_uint8)

        return rgb_uint8

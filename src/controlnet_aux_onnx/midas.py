import numpy as np
from .detector import Detector


class MidasDetector(Detector):
    def __init__(self, model_dir="./models", providers=None, provider_options=None, sess_options=None):
        super().__init__("midas",
                         model_dir=model_dir,
                         providers=providers,
                         provider_options=provider_options,
                         sess_options=sess_options)
        self.divisible_factor = 32

        self.postprocessor_name = "midas_postprocessor_normal"
        self.models[self.postprocessor_name] = self._load_model(self.postprocessor_name, model_dir, providers,
                                                                provider_options,sess_options)

    def _infer(self, models, model_name, input_data, **kwargs):
        depth_and_normal = kwargs.get("depth_and_normal")

        session, input_names, output_names = models[model_name]
        if depth_and_normal:
            output_names = None
        else:
            output_names = [output_names[0]]

        onnx_input = {input_names[0]: input_data}

        if depth_and_normal:
            depth_output, depth = session.run(output_names, onnx_input)
            postprocessor, post_input_names, post_output_names = models[self.postprocessor_name]
            a = kwargs.get("a")
            a = np.pi * 2.0 if a is None else a
            bg_th = kwargs.get("bg_th")
            bg_th = 0.1 if bg_th is None else bg_th

            normal_output = postprocessor.run(post_output_names, {post_input_names[0]: depth,
                                                                  post_input_names[1]: np.array([a], dtype=np.float32),
                                                                  post_input_names[2]: np.array([bg_th], dtype=np.float32)})[0]
            return depth_output, normal_output

        depth_output = session.run(output_names, onnx_input)[0]
        return depth_output


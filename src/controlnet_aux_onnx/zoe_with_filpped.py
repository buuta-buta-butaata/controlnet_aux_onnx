import numpy as np
from .detector import Detector


class ZoeDetector(Detector):
    def __init__(self, model_dir="./models", providers=None, provider_options=None, sess_options=None):
        super().__init__("zoe_inference",
                         model_dir=model_dir,
                         providers=providers,
                         provider_options=provider_options,
                         sess_options=sess_options)
        self.divisible_factor = 32

        self.preprocessor_name = "zoe_preprocessor"
        self.postprocessor_name = "zoe_postprocessor"
        self.models[self.preprocessor_name] = self._load_model(self.preprocessor_name, model_dir, providers,
                                                               provider_options, sess_options)
        self.models[self.postprocessor_name] = self._load_model(self.postprocessor_name, model_dir, providers,
                                                                provider_options, sess_options)

        self.filter_gamma_corrected_name = "filter_gamma_corrected"
        self.models[self.filter_gamma_corrected_name] = self._load_model(self.filter_gamma_corrected_name, model_dir,
                                                                         providers, provider_options, sess_options)

    def _infer(self, models, model_name, input_data, **kwargs):
        preprocessor, pre_input_names, pre_output_names = models[self.preprocessor_name]
        session, input_names, output_names = models[model_name]
        postprocessor, post_input_names, post_output_names = models[self.postprocessor_name]

        normalized_image, normalized_image_flipped = preprocessor.run(pre_output_names,
                                                                      {pre_input_names[0]: input_data})

        onnx_input = {input_names[0]: normalized_image}
        depth = session.run(output_names, onnx_input)[0]

        onnx_input_flipped = {input_names[0]: normalized_image_flipped}
        depth_flipped = session.run(output_names, onnx_input_flipped)[0]

        output = postprocessor.run(post_output_names, {post_input_names[0]: depth,
                                                       post_input_names[1]: depth_flipped})[0]
        return output

    def _filter(self, image, **kwargs):
        gamma_corrected = kwargs.get("gamma_corrected")

        if gamma_corrected:
            sess, in_names, out_names = self.models[self.filter_gamma_corrected_name]
            image = sess.run(None, {in_names[0]: image})[0]
        return image


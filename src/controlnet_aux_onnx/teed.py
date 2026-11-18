import numpy as np

from .detector import Detector


class TEEDdetector(Detector):
    def __init__(self, model_dir="./models", providers=None, provider_options=None, sess_options=None):
        super().__init__("teed",
                         model_dir=model_dir,
                         providers=providers,
                         provider_options=provider_options,
                         sess_options=sess_options)
        self.safe_step_name = "filter_safe_step"
        self.models[self.safe_step_name] = self._load_model(self.safe_step_name, model_dir, providers, provider_options,
                                                            sess_options)

    # controlnet_auxの実装と合わせるため、image_resolutionを除外しておく
    def __call__(
            self,
            input_image,
            detect_resolution=512,
            output_type="pil",
            **kwargs):
        if kwargs.get("image_resolution"):
            del kwargs["image_resolution"]

        return super().__call__(input_image=input_image,
                                detect_resolution=detect_resolution,
                                image_resolution=0,
                                output_type=output_type,
                                **kwargs)

    def _filter(self, image, **kwargs):
        safe_steps = kwargs.get("safe_steps")

        if safe_steps is not None and safe_steps != 0:
            sess, in_names, out_names = self.models[self.safe_step_name]
            image = sess.run(None, {in_names[0]: image, in_names[1]: np.array([safe_steps])})[0]
        return image.clip(0.0, 1.0)


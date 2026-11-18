import numpy as np
from .detector import Detector


class HEDdetector(Detector):
    def __init__(self, model_dir="./models", providers=None, provider_options=None, sess_options=None):
        super().__init__("hed",
                         model_dir=model_dir,
                         providers=providers,
                         provider_options=provider_options,
                         sess_options=sess_options)
        self.safe_step_name = "filter_safe_step"
        self.scribble_name = "filter_scribble"
        self.models[self.safe_step_name] = self._load_model(self.safe_step_name, model_dir, providers, provider_options,
                                                            sess_options)
        self.models[self.scribble_name] = self._load_model(self.scribble_name, model_dir, providers, provider_options,
                                                           sess_options)

    def _filter(self, image, **kwargs):
        safe = kwargs.get("safe")
        scribble = kwargs.get("scribble")

        if safe:
            sess, in_names, out_names = self.models[self.safe_step_name]
            image = sess.run(None, {in_names[0]: image, in_names[1]: np.array([2])})[0]
        if scribble:
            sess, in_names, out_names = self.models[self.scribble_name]
            image = sess.run(None, {in_names[0]: image.clip(0.0, 1.0)})[0]
        return image.clip(0.0, 1.0)

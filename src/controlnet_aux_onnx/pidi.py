import numpy as np
from .detector import Detector


class PidiNetDetector(Detector):
    def __init__(self, model_dir="./models", providers=None, provider_options=None, sess_options=None):
        super().__init__("pidi",
                         model_dir=model_dir,
                         providers=providers,
                         provider_options=provider_options,
                         sess_options=sess_options)

        self.filter_pidi_name = "filter_pidi"
        self.safe_step_name = "filter_safe_step"
        self.scribble_name = "filter_scribble"
        self.models[self.filter_pidi_name] = self._load_model(self.filter_pidi_name, model_dir, providers,
                                                              provider_options, sess_options)
        self.models[self.safe_step_name] = self._load_model(self.safe_step_name, model_dir, providers, provider_options,
                                                            sess_options)
        self.models[self.scribble_name] = self._load_model(self.scribble_name, model_dir, providers, provider_options,
                                                           sess_options)

    def _filter(self, image, **kwargs):
        apply_filter = kwargs.get("apply_filter")
        safe = kwargs.get("safe")
        scribble = kwargs.get("scribble")

        if apply_filter:
            # image = image > 0.5
            sess, in_names, out_names = self.models[self.filter_pidi_name]
            image = sess.run(None, {in_names[0]: image})[0]
        if safe:
            # image = safe_step(image)
            sess, in_names, out_names = self.models[self.safe_step_name]
            image = sess.run(None, {in_names[0]: image, in_names[1]: np.array([2])})[0]
        if scribble:
            sess, in_names, out_names = self.models[self.scribble_name]
            image = sess.run(None, {in_names[0]: image.clip(0.0, 1.0)})[0]
        return image.clip(0.0, 1.0)


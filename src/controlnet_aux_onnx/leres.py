import numpy as np
from .detector import Detector


class LeresDetector(Detector):
    def __init__(self, model_dir="./models", providers=None, provider_options=None, sess_options=None):
        super().__init__("leres",
                         model_dir=model_dir,
                         providers=providers,
                         provider_options=provider_options,
                         sess_options=sess_options)
        self.remove_near_name = "filter_remove_near"
        self.remove_background_name = "filter_remove_background"
        self.models[self.remove_near_name] = self._load_model(self.remove_near_name, model_dir, providers,
                                                              provider_options,sess_options)
        self.models[self.remove_background_name] = self._load_model(self.remove_background_name,
                                                                    model_dir, providers, provider_options, sess_options)

    def _filter(self, image, **kwargs):
        # remove near
        thr_a = kwargs.get("thr_a")

        if thr_a is not None and thr_a != 0:
            sess, in_names, out_names = self.models[self.remove_near_name]
            image = sess.run(None, {in_names[0]: image, in_names[1]: np.array([thr_a])})[0]

        # remove background
        thr_b = kwargs.get("thr_b")

        if thr_b is not None and thr_b != 0:
            sess, in_names, out_names = self.models[self.remove_background_name]
            image = sess.run(None, {in_names[0]: image, in_names[1]: np.array([thr_b])})[0]

        return image

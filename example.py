import numpy as np
from PIL import Image
from controlnet_aux_onnx import PidiNetDetector, TEEDdetector, HEDdetector

img_path = r"./tests/test_images/ninja512.png"
model_dir = "./models"
# CPU
ep_list = ["CPUExecutionProvider"]
# GPU
# ep_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]

img = Image.open(img_path).convert("RGB")

# PidiNet
pidi = PidiNetDetector(model_dir=model_dir, providers=ep_list)
pidi_img = pidi(img)
pidi_img.save("pidinet_output.png")

pidi_scribble_img = pidi(img, scribble=True)
pidi_scribble_img.save("pidinet_scribble_output.png")

# TEED
teed = TEEDdetector(model_dir=model_dir, providers=ep_list)
teed_img = teed(img)
teed_img.save("teed_output.png")

# HED
hed = HEDdetector(model_dir=model_dir, providers=ep_list)
hed_img = hed(img)
hed_img.save("hed_output.png")

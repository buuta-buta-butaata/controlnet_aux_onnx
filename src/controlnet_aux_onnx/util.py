# From: https://github.com/huggingface/controlnet_aux/blob/master/src/controlnet_aux/util.py
# import cv2
import numpy as np
from PIL import Image

# Modified: Add divisible_factor, Remove dependency on cv2
def resize_image(input_image, resolution, divisible_factor = 64):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / divisible_factor)) * divisible_factor
    W = int(np.round(W / divisible_factor)) * divisible_factor
    # img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    pil_img = Image.fromarray(input_image).resize((W, H), Image.Resampling.LANCZOS if k > 1 else Image.Resampling.BOX)
    img = np.array(pil_img)
    return img

def resize_to(input_image, W, H):
    pil_img = Image.fromarray(input_image).resize((W, H), Image.Resampling.BILINEAR)
    img = np.array(pil_img)
    return img

import os
from statistics import mean

import cv2
import numpy as np

from common.debug_image import debug_image
from helpers.utils import read_panels
from scipy import stats as st


def filter_ratio(c):
    _, _, w, h = cv2.boundingRect(c)
    return 0.2 < w / h < 1.2


def get_h(c):
    _, _, _, h = cv2.boundingRect(c)
    return h

def get_w(c):
    _, _, w, _ = cv2.boundingRect(c)
    return w


base_path = "resources/test_ocr_panels/"
panel_names = list(filter(lambda p: 'png' in p or 'PNG' in p, os.listdir(base_path)))
panel_images_bgr = list(map(lambda p: read_panels(base_path, p), panel_names))
for image_bgr, name in zip(panel_images_bgr, panel_names):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_gray = 255 - image_gray
    mask = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 61, 23)
    debug_image(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #contours = list(filter(filter_ratio, contours))
    #mean = sum(list(map(get_h, contours))) / len(contours)
    mode_h, _ = st.mode(np.array(list(map(get_h, contours))))
    mode_w, _ = st.mode(np.array(list(map(get_w, contours))))
    for c in contours:
        copia = image_rgb.copy()
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(copia, (x, y), (x + w, y + h), (255, 0, 0), 1)
        if mode_h - 3 < h < mode_h + 3 and mode_w - 3 < w < mode_w + 3:
            cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 1)
    print(name)
    debug_image(copia)
    debug_image(image_rgb)

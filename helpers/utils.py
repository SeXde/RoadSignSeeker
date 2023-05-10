import os
from pathlib import Path

import cv2
import numpy as np

from common.poi import Poi
from helpers.paths import GENERATED_IMG_PATH


def save_panels(img_path: str, pois: [Poi]):
    image = cv2.imread(img_path)
    for poi in pois:
        write_rectangles(image, poi)
    cv2.imwrite("{}/{}".format(GENERATED_IMG_PATH, Path(img_path).name), image)


def write_rectangles(image, poi: Poi, rect_color=(0, 0, 255)):
    shape = poi.shape
    xl, yl = (shape.x, shape.y)
    cv2.rectangle(image, (shape.x, shape.y), (shape.x + shape.w, shape.y + shape.h), rect_color, 2)
    cv2.putText(image, str(round(poi.score, 2)), (xl, yl + shape.h), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 233, 255), 2)


def draw_classes(pois: [Poi], classes: [int]) -> None:
    available_classes = np.unique(classes)
    max_colors = len(available_classes)

    colors = generate_random_colors(max_colors)

    for poi_index in range(len(pois)):
        poi = pois[poi_index]
        c = classes[poi_index]

        class_index = available_classes.tolist().index(c)
        color = hsv_to_rgb(colors[class_index] / 180, 1, 1)
        poi.c = [color[0] * 255, color[1] * 255, color[2] * 255]


def generate_random_colors(max_colors: int) -> []:
    return [(180 / max_colors) * i for i in range(0, max_colors)]


def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v

    i = int(h*6.)  # XXX assume int() truncates!
    f = (h*6.)-i
    p, q, t = v * (1. - s), v * (1. - s * f), v * (1. - s * (1. - f))
    i %= 6

    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q
from pathlib import Path

import cv2

from common.poi import Poi
from roadSeekerIo.paths import GENERATED_IMG_PATH


def save_panels(img_path: str, pois: [Poi]):
    image = cv2.imread(img_path)
    map(lambda poi: write_rectangles(image, poi), pois)
    cv2.imwrite("{}/{}".format(GENERATED_IMG_PATH, Path(img_path).name), image)


def write_rectangles(image, poi: Poi):
    shape = poi.shape
    xl, yl = (shape.x, shape.y)
    cv2.rectangle(image, (shape.x, shape.y), (shape.x + shape.w, shape.y + shape.h), (255, 0, 0), 2)
    cv2.putText(image, str(round(poi.score, 2)), (xl, yl + shape.h), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 233, 0), 2)

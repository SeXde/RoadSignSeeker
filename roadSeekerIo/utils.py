from pathlib import Path

import cv2

from common.poi import Poi
from roadSeekerIo.paths import GENERATED_IMG_PATH


def save_panels(img_path: str, pois: [Poi]):
    image = cv2.imread(img_path)
    map(lambda poi: _write_rectangles(image, poi), pois)
    cv2.imwrite("{}/{}".format(GENERATED_IMG_PATH, Path(img_path).name), image)


def _write_rectangles(image, poi: Poi):
    xl, yl = poi.get_lower_edge()
    cv2.rectangle(image, poi.get_upper_edge(), (xl, yl), (0, 0, 255), 2)
    cv2.putText(image, str(poi.score), (xl, yl - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 233, 255), 1)

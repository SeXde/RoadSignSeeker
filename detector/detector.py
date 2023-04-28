import cv2

from common.poi import Poi
from common.shape import Shape


class Detector:
    def detect(self, img_path: str) -> [Poi]:
        pass

    @staticmethod
    def _region_to_poi(region, i_rgb, img_path) -> Poi:
        x, y, w, h = cv2.boundingRect(region)
        return Poi(Shape(x, y, w, h), i_rgb, img_path)

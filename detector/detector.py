import cv2

from common.poi import Poi
from common.shape import Shape


class Detector:
    def detect(self, img_path: str) -> [Poi]:
        pass

    @staticmethod
    def _region_to_poi(region, i_rgb, img_path) -> Poi:
        x, y, w, h = cv2.boundingRect(region)

        extra_region = 2

        shape = Shape(x=max(x - extra_region, 0),
                      y=max(y - extra_region, 0),
                      width=w + extra_region * 2,
                      height=h + extra_region * 2)

        return Poi(shape, i_rgb, img_path)

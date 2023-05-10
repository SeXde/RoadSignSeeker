import cv2

from common.poi import Poi
from common.shape import Shape


class Detector:
    def detect(self, img_path: str) -> [Poi]:
        pass

    @staticmethod
    def _region_to_poi(region, i_rgb, img_path, extra_space=0) -> Poi:
        x, y, w, h = cv2.boundingRect(region)

        shape = Shape(x=max(x - extra_space, 0),
                      y=max(y - extra_space, 0),
                      width=min(w + extra_space * 2, i_rgb.shape[0]),
                      height=min(h + extra_space * 2, i_rgb.shape[1]))

        return Poi(shape, i_rgb, img_path)

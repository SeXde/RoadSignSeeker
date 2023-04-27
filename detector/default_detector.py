import cv2
import numpy as np

from common.poi import Poi
from common.shape import Shape


class DefaultDetector:

    def detect(self, img_path: str) -> [Poi]:
        i_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        i_rgb = cv2.cvtColor(i_bgr, cv2.COLOR_BGR2RGB)
        i_gray = cv2.cvtColor(i_rgb, cv2.COLOR_RGB2GRAY)
        i_gray = cv2.equalizeHist(cv2.equalizeHist(i_gray))
        mser = cv2.MSER_create(max_area=99000)
        regions, _ = mser.detectRegions(i_gray)
        return list(map(lambda r: self._region_to_poi(r, i_rgb, img_path), regions))

    @staticmethod
    def _region_to_poi(region, i_rgb, img_path) -> Poi:
        x, y, w, h = cv2.boundingRect(region)
        return Poi(Shape(x, y, w, h), i_rgb, img_path)

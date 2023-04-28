import cv2
import numpy as np

from common.debug_image import debug_image
from common.poi import Poi
from common.shape import Shape
from detector.detector import Detector


class ContourDetector(Detector):
    def detect(self, img_path: str) -> [Poi]:
        i_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        i_rgb = cv2.cvtColor(i_bgr, cv2.COLOR_BGR2RGB)
        i_hsv = cv2.cvtColor(i_bgr, cv2.COLOR_BGR2HSV)

        mask_image = cv2.inRange(i_hsv, np.array([100, 150, 0]), np.array([140, 255, 255]))

        contour = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return list(map(lambda r: self._region_to_poi(r, i_rgb, img_path), contour[0]))

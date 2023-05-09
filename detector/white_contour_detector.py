import cv2
import numpy as np

from common.debug_image import debug_image
from common.poi import Poi
from common.shape import Shape
from detector.contour_detector import ContourDetector
from detector.detector import Detector


class WhiteContourDetector(ContourDetector):
    def __init__(self, low_color: (int, int, int), high_color: (int, int, int)):
        super().__init__(low_color, high_color)

    def detect(self, img_path: str) -> [Poi]:
        i_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        i_rgb = cv2.cvtColor(i_bgr, cv2.COLOR_BGR2RGB)

        alpha = 3  # Contrast control (1.0-3.0)
        beta = 0  # Brightness control (0-100)

        adjusted = cv2.convertScaleAbs(i_rgb, alpha=alpha, beta=beta)
        adjusted_hsv = cv2.cvtColor(adjusted, cv2.COLOR_RGB2HSV)
        mask_image = cv2.inRange(adjusted_hsv, self.low_color, self.high_color)

        print(self.low_color, self.high_color)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        mask_image_dilated = cv2.dilate(mask_image, kernel)

        debug_image(mask_image)
        debug_image(mask_image_dilated)

        contour = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return list(map(lambda r: self._region_to_poi(r, i_rgb, img_path, extra_space=2), contour[0]))

import cv2

from common.poi import Poi
from detector.detector import Detector


class ContourDetector(Detector):
    """
    Detects Pois using findContours.
    FindContours always returns non-overlapping Pois
    Constructor params allow to generate the contours using different color masks
    """
    def __init__(self, low_color: (int, int, int), high_color: (int, int, int)):
        self.low_color = low_color
        self.high_color = high_color

    def detect(self, img_path: str) -> [Poi]:
        i_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        i_rgb = cv2.cvtColor(i_bgr, cv2.COLOR_BGR2RGB)
        i_hsv = cv2.cvtColor(i_bgr, cv2.COLOR_BGR2HSV)

        mask_image = cv2.inRange(i_hsv, self.low_color, self.high_color)

        contour = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return list(map(lambda r: self._region_to_poi(r, i_rgb, img_path, extra_space=2), contour[0]))

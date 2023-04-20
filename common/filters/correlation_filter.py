import cv2
import numpy as np

from common.filter import Filter
from common.poi import Poi


class CorrelationFilter(Filter):

    def __init__(self, limit : float, mask_image : [int, int, int], lower_color : [int], upper_color : [int]):
        self.limit = limit
        self.mask_image = mask_image
        self.upper_color = np.array(upper_color)
        self.lower_color = np.array(lower_color)

    def apply(self, poi: Poi) -> bool:
        hsv_rect = poi.get_hsv_rect()
        hsv_rect_resized = cv2.resize(hsv_rect, (80, 40))
        hsv_rect_resized = cv2.inRange(hsv_rect_resized, self.lower_color, self.upper_color)
        corr_arr = cv2.matchTemplate(hsv_rect_resized, self.mask_image, cv2.TM_CCORR_NORMED)
        corr = corr_arr[0, 0]
        poi.score = corr

        return corr > self.limit

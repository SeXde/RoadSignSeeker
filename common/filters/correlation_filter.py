import cv2
import numpy as np

from common.debug_image import debug_image
from common.filter import Filter
from common.poi import Poi


class CorrelationFilter(Filter):
    """
    This filter correlates the Poi with a mask, this operation generates a score.
    If the score is below the limit constructor param, the Poi is filtered out.
    """
    def __init__(self, limit: float, mask_image: [int, int, int], lower_color: (int, int, int), upper_color: (int, int, int)):
        self.limit = limit
        self.mask_image = mask_image
        self.upper_color = upper_color
        self.lower_color = lower_color

    def apply(self, poi: Poi) -> bool:
        hsv_rect = poi.get_hsv_rect()

        mask = cv2.inRange(hsv_rect, self.lower_color, self.upper_color)

        resized_mask_image = cv2.resize(self.mask_image, (hsv_rect.shape[1], hsv_rect.shape[0]))

        corr_arr = cv2.matchTemplate(mask, resized_mask_image, method=cv2.TM_CCORR_NORMED)
        corr = corr_arr[0, 0]
        poi.score = corr

        return corr > self.limit

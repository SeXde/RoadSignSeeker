import cv2
import numpy as np

from common.filter import Filter
from common.poi import Poi


class CorrelationFilter(Filter):
    def __init__(self, limit: float, mask_image: [int, int, int], lower_color: [int], upper_color: [int]):
        self.limit = limit
        self.mask_image = mask_image
        self.upper_color = np.array(upper_color)
        self.lower_color = np.array(lower_color)

    def apply(self, poi: Poi) -> bool:
        hsv_rect = poi.get_hsv_rect()
        mask = cv2.inRange(hsv_rect, self.lower_color, self.upper_color)
        mask = cv2.resize(mask, self.mask_image.shape)

        corr_arr = cv2.matchTemplate(self.mask_image, mask, cv2.TM_CCORR_NORMED)
        corr = corr_arr[0, 0]

        poi.score = round(corr, 2)

        return corr > self.limit

import cv2
from common.shape import Shape


class Poi:
    def __init__(self, shape: Shape, full_rgb_image: [int]):
        self.shape = shape

        self._rgb_image = self._get_rect(full_rgb_image)

        self._loaded_hsv = False
        self._hsv_image = None

        self._loaded_gray = False
        self._gray_image = None

    def get_hsv_rect(self) -> [int, int, int]:
        if self._loaded_hsv:
            return self._hsv_image

        self._hsv_image = cv2.cvtColor(self._rgb_image, cv2.COLOR_RGB2HSV_FULL)
        self._loaded_hsv = True

        return self._hsv_image

    def get_gray_rect(self) -> [int, int, int]:
        if self._loaded_gray:
            return self._gray_image

        self._gray_image = cv2.cvtColor(self._rgb_image, cv2.COLOR_RGB2GRAY)
        self._gray_image = cv2.equalizeHist(self._gray_image)

        self._loaded_gray = True

        return  self._gray_image

    def get_rgb_rect(self) -> [int, int, int]:
        return self._rgb_image

    def _get_rect(self, img: [int, int, int]) -> [int, int, int]:
        return img[
           self.shape.y:self.shape.y + self.shape.h,
           self.shape.x:self.shape.x + self.shape.w
       ]

    def apply_filters(self, filters: []) -> bool:
        for fil in filters:
            if not fil.apply(self):
                return False

        return True

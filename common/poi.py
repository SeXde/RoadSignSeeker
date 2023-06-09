from pathlib import Path

import cv2
from common.shape import Shape
from helpers.paths import RESULT_FILE_NAME


class Poi:
    """
    Class that represents a Point Of Interest in an image.
    Also allows to operate in the image, like getting the image in RGB, HSV, grayscale...
    Getting the image in different color schemes is a lazy process,
    which means that different color schemes are not processed until requested.
    """
    def __init__(self, shape: Shape, full_rgb_image: [int], file_path: str):
        self.shape = shape

        self._rgb_image = self._get_rect(full_rgb_image)

        self._loaded_hsv = False
        self._hsv_image = None

        self._loaded_gray = False
        self._gray_image = None
        self.panel_type = 1
        self.score = 0
        self.file_path = file_path
        self.file_name = Path(file_path).name
        self.c = -1

    def get_hsv_rect(self) -> [int, int, int]:
        if self._loaded_hsv:
            return self._hsv_image

        self._hsv_image = cv2.cvtColor(self._rgb_image, cv2.COLOR_RGB2HSV)
        self._loaded_hsv = True

        return self._hsv_image

    def get_gray_rect(self) -> [int, int, int]:
        if self._loaded_gray:
            return self._gray_image

        self._gray_image = cv2.cvtColor(self._rgb_image, cv2.COLOR_RGB2GRAY)
        self._gray_image = cv2.equalizeHist(self._gray_image)

        self._loaded_gray = True

        return self._gray_image

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

    def _to_string(self) -> str:
        return "{};{};{};{};{};{};{}".format(
            self.file_name,
            self.shape.x,
            self.shape.y,
            self.shape.x + self.shape.w,
            self.shape.y + self.shape.h,
            self.panel_type,
            "%.2f" % round(self.score, 2)
        ) + '\n'

    def save_to_file(self):
        with open(RESULT_FILE_NAME, 'a+') as f:
            f.write(self._to_string())


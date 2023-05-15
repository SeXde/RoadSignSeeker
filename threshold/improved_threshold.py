import cv2

from common.debug_image import debug_image
from threshold.threshold import Threshold


class ImprovedThreshold(Threshold):
    def __init__(self, dilation_kernel: (int, int)):
        self.dilation_kernel = dilation_kernel

    def threshold_image(self, gray_img: [int], invert: bool = False) -> []:
        mask = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 199, 5)

        mask = 255 - mask

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        mask = cv2.dilate(mask, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    def threshold_image_class(self, gray_img: [int], invert: bool = False) -> []:
        mask = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 199, 5)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.dilation_kernel)

        mask = cv2.dilate(mask, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP , cv2.CHAIN_APPROX_SIMPLE)

        contours = list(filter(self.filter_contours, contours))

        return contours, mask

    def filter_contours(self, c) -> bool:
        area = cv2.contourArea(c)

        _, _, w, h = cv2.boundingRect(c)
        ratio = w / h

        return 50 < area < 850 and 0.3 < ratio < 3


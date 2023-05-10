import cv2

from threshold.threshold import Threshold


class DefaultThreshold(Threshold):

    def threshold_image(self, gray_img: [int], invert: bool = False) -> []:
        mask = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 199, 5)
        if invert:
            mask = 255 - mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

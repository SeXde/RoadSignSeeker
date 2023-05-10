import cv2

from threshold.threshold import Threshold


class ImprovedThreshold(Threshold):

    def threshold_image(self, gray_img: [int]) -> []:
        mask = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 199, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        mask = cv2.dilate(mask, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

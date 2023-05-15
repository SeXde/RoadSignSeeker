import cv2

from threshold.threshold import Threshold


class DefaultThreshold(Threshold):

    def threshold_image(self, gray_img: [int], invert: bool = False) -> []:
        contours, _ = self.threshold_image_class(gray_img, invert)

        return contours

    def threshold_image_class(self, gray_img: [int], invert: bool = False) -> []:
        mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        contour = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contour = contour[0] if len(contour) == 2 else contour[1]
        return list(filter(self.area_filter, contour)), mask

    def area_filter(self, c) -> bool:
        return cv2.contourArea(c) < 850 and cv2.contourArea(c) > 50

import cv2
import numpy as np
from skimage.measure import LineModelND, ransac


class LineDetector:

    def get_centers(self, contours):
        cx = []
        cy = []
        for c in contours:
            # compute the center of the contour
            M = cv2.moments(c)
            if M['m00'] != 0:
                cx.append(float(M["m10"] / M["m00"]))
                cy.append(float(M["m01"] / M["m00"]))
        return np.array(cx), np.array(cy)

    def get_x(self, contour):
        x, _, _, _ = cv2.boundingRect(contour)
        return x

    def get_y(self, contour):
        _, y, _, _ = cv2.boundingRect(contour)
        return y

    def detect_lines(self, contours, image_rgb) -> []:
        lines = []
        contours = list(filter(lambda c: cv2.moments(c)['m00'] != 0, contours))
        found_lines = False
        while not found_lines:
            cx, cy = self.get_centers(contours)
            if len(lines) >= 5:
                break
            if len(cx) <= 2:
                break
            model, inliers = ransac(np.stack([cx, cy], axis=-1), LineModelND, min_samples=2, residual_threshold=2,
                                         max_trials=1000)
            if inliers is None:
                break
            line = []
            new_contours = []
            sy, ey = (0, 200)
            sx, ex = model.predict_x([sy, ey])
            cv2.line(image_rgb, (round(sx), sy), (round(ex), ey), (225, 242, 70), 1)
            for i in range(len(inliers)):
                if inliers[i]:
                    line.append(contours[i])
                else:
                    new_contours.append(contours[i])
            if new_contours == contours:
                break
            contours = new_contours
            line = sorted(line, key=lambda c: self.get_x(c))
            lines.append(line)
        lines = sorted(lines, key=lambda l: self.get_y(l[0]))
        return lines

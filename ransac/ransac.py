import cv2
import numpy as np
from skimage.measure import LineModelND, ransac


def get_centers(contours):
    cx = []
    cy = []
    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx.append(float(M["m10"] / M["m00"]))
            cy.append(float(M["m01"] / M["m00"]))
    return np.array(cx), np.array(cy)


def get_x(contour):
    x, _, _, _ = cv2.boundingRect(contour)
    return x

def get_y(contour):
    _, y, _, _ = cv2.boundingRect(contour)
    return y


def detect_lines(contours) -> []:
    lines = []
    contours = list(filter(lambda c: cv2.moments(c)['m00'] != 0, contours))
    found_lines = False
    while not found_lines:
        cx, cy = get_centers(contours)
        _, inlines = ransac(np.stack([cx, cy], axis=-1), LineModelND, min_samples=3, residual_threshold=2, max_trials=1000)
        line = []
        new_contours = []
        for i in range(len(inlines)):
            if inlines[i]:
                line.append(contours[i])
            else:
                new_contours.append(contours[i])
        contours = new_contours
        line = sorted(line, key=lambda c: get_x(c))
        lines.append(line)
        found_lines = len(lines) > 4 or len(inlines) == 0
    lines = sorted(lines, key=lambda l: get_y(l[0]))
    return lines

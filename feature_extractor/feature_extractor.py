from random import seed, random

import cv2
import numpy as np

from common.debug_image import debug_image


class FeatureExtractor:
    """
    This class is able to generate a characteristics array from an image and some regions.
    """
    def extract(self, contours, class_id, image, C, E):
        for cont in contours:
            x, y, w, h = cv2.boundingRect(cont)

            max_size = h if h > w else w
            min_size = w if h > w else h

            diff = max_size - min_size
            if diff % 2 != 0:
                max_size += 1
                diff += 1

            im_rect = image[y:y + h, x:x + w]

            v_pad = (0,) if h > w else (diff // 2,)
            h_pad = (diff // 2,) if h > w else (0,)

            padded_im = np.pad(im_rect, [v_pad, h_pad], mode='constant', constant_values=(255, 255))

            resized_im = cv2.resize(padded_im, (25, 25))

            c = np.array(resized_im.flatten())

            C.append(c)
            E.append(class_id)
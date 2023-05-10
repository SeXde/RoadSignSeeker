import numpy as np

from common.filter import Filter
from common.poi import Poi


class DominantColorFilter(Filter):
    """
    This implementation gets the dominant color of an image, and then checks it against the given color params.
    This filter is not recommended as it filters too many valid Pois.
    """
    def __init__(self, hue: (int | None, int | None) = (None, None),
                 saturation: (int | None, int | None) = (None, None),
                 value: (int | None, int | None) = (None, None)
                 ):
        # For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
        self._hue = (hue[0] or 0, hue[1] or 179)
        self._saturation = (saturation[0] or 0, saturation[1] or 255)
        self._value = (value[0] or 0, value[1] or 255)

    def apply(self, poi: Poi) -> bool:
        dominant_color = self.get_dominant_color(poi.get_hsv_rect())

        return self._hue[0] < dominant_color[0] < self._hue[1] \
            and self._saturation[0] < dominant_color[1] < self._saturation[1] \
            and self._value[0] < dominant_color[2] < self._value[1]

    @staticmethod
    def get_dominant_color(img) -> [int, int, int]:
        colors, count = np.unique(img.copy().reshape(-1, img.shape[-1]), axis=0, return_counts=True)
        return colors[count.argmax()]

import numpy as np

from common.filter import Filter
from common.poi import Poi


class DominantColorFilter(Filter):
    def __init__(self, min_hue: int, max_hue: int):
        self.min_blue = min_hue
        self.max_blue = max_hue

    def apply(self, poi: Poi) -> bool:
        dominant_color = self.get_dominant_color(poi.get_hsv_rect())

        # TODO: Parametrize the dominant_color[2] in constructor
        return self.min_blue < dominant_color[0] < self.max_blue and dominant_color[2] > 150

    @staticmethod
    def get_dominant_color(img) -> [int, int, int]:
        colors, count = np.unique(img.copy().reshape(-1, img.shape[-1]), axis=0, return_counts=True)
        return colors[count.argmax()]

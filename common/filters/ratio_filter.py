from common.filter import Filter
from common.poi import Poi


class RatioFilter(Filter):
    def __init__(self, min_ratio: float = 0, max_ratio: float = 1.5):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def apply(self, poi: Poi) -> bool:
        ratio = poi.shape.w / poi.shape.h

        return self.min_ratio < ratio < self.max_ratio

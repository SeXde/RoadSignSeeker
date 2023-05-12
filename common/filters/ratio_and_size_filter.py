from common.filter import Filter
from common.poi import Poi


class RatioAndSizeFilter(Filter):
    """
    Filters the Poi by ratio and by width
    """
    def __init__(self, min_ratio: float = 0, max_ratio: float = 1.5, max_width: int = 90):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.max_width = max_width

    def apply(self, poi: Poi) -> bool:
        ratio = poi.shape.w / poi.shape.h
        return self.min_ratio < ratio < self.max_ratio and poi.shape.w > self.max_width

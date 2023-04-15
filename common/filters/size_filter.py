from common.filter import Filter
from common.poi import Poi


class SizeFilter(Filter):
    def __init__(self, width: (int | None, int | None), height: (int | None, int | None)):
        self.min_width = width[0]
        self.max_width = width[1]

        self.min_height = height[0]
        self.max_height = height[1]

    def apply(self, poi: Poi) -> bool:
        shape = poi.shape

        is_min_width = self.min_width < shape.w if self.min_width is not None else True
        is_max_width = shape.w < self.max_width if self.max_width is not None else True

        is_min_height = self.min_height < shape.h if self.min_height is not None else True
        is_max_height = shape.h < self.max_height if self.max_height is not None else True

        return is_min_width and is_min_height and is_max_width and is_max_height

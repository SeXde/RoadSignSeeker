import numpy as np

from common.filter import apply_filters
from common.filters.correlation_filter import CorrelationFilter
from common.filters.ratio_and_size_filter import RatioAndSizeFilter
from common.poi import Poi
from pipelines.pipeline import Pipeline
from helpers.colors import HIGH_BLUE, LOW_BLUE


class NoOverlapPipeline(Pipeline):
    def __init__(self):
        self.img_mask = np.full((1000, 1000), 255, dtype=np.uint8)

    def apply(self, pois: [Poi]) -> [Poi]:
        valid_pois = []

        for poi in pois:
            res = apply_filters(poi, [
                RatioAndSizeFilter(0.8, 3.92, 40),
                CorrelationFilter(0.7, self.img_mask, LOW_BLUE, HIGH_BLUE)
            ])

            if res:
                valid_pois.append(poi)

        return valid_pois

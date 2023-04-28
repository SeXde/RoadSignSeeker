import numpy as np

from common.filter import apply_filters
from common.filters.correlation_filter import CorrelationFilter
from common.filters.nms_filter import NMSMultiFilter
from common.filters.ratio_and_size_filter import RatioAndSizeFilter
from common.multifilter import apply_multifilters
from common.poi import Poi
from pipelines.pipeline import Pipeline


class NMSPipeline(Pipeline):
    def __init__(self):
        self.img_mask = np.full((1000, 1000), 255, dtype=np.uint8)

    def apply(self, pois: [Poi]) -> [Poi]:
        valid_pois = []

        for poi in pois:
            res = apply_filters(poi, [
                RatioAndSizeFilter(0.8, 3.92),
                CorrelationFilter(0.8, self.img_mask, [100, 150, 0], [140, 255, 255])
            ])

            if res:
                valid_pois.append(poi)

        valid_pois = apply_multifilters(valid_pois, [NMSMultiFilter()])

        return valid_pois

import cv2
import numpy as np

from common.filter import apply_filters
from common.filters.aabb_filter import AabbMultiFilter
from common.filters.ratio_and_size_filter import RatioAndSizeFilter
from common.filters.correlation_filter import CorrelationFilter
from common.multifilter import apply_multifilters
from common.poi import Poi


def default_pipeline(pois: [Poi]) -> [Poi]:
    perfect_mask = np.full((1000, 1000), 255, dtype=np.uint8)

    valid_pois = []

    for poi in pois:
        res = apply_filters(poi, [
            RatioAndSizeFilter(0.8, 3.92),
            CorrelationFilter(0.8, perfect_mask, [90, 100, 20], [130, 255, 255])
        ])

        if res:
            valid_pois.append(poi)

    #valid_pois = apply_multifilters(valid_pois, [AabbMultiFilter()])

    for image in valid_pois:
        #debug_image(image.get_rgb_rect())
        pass

    return valid_pois

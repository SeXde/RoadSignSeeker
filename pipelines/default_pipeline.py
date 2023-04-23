import cv2

from common.debug_image import debug_image
from common.filter import apply_filters
from common.filters.aabb_filter import AabbMultiFilter
from common.filters.correlation_filter import CorrelationFilter
from common.filters.dominant_color_filter import DominantColorFilter
from common.filters.ratio_filter import RatioFilter
from common.filters.size_filter import SizeFilter
from common.multifilter import apply_multifilters
from common.poi import Poi


def default_pipeline(pois: [Poi]) -> [Poi]:
    mask_path = "resources/RoadSign_Mask.png"
    img_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    valid_pois = []

    for poi in pois:
        res = apply_filters(poi, [
            #SizeFilter(width=(50, None), height=(0, None)),
            #RatioFilter(1, 3.92),
            #DominantColorFilter(hue=(150, 200), value=(150, None))
            CorrelationFilter(0.5, img_mask, [200 / 2, 200, 200], [250 / 2, 255, 255])
        ])

        if res:
            valid_pois.append(poi)

    valid_pois = apply_multifilters(valid_pois, [AabbMultiFilter()])

    for image in valid_pois:
        #debug_image(image.get_rgb_rect())
        pass

    return valid_pois

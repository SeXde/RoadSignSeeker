import numpy as np

from common.multifilter import MultiFilter
from common.poi import Poi


class NMSMultiFilter(MultiFilter):
    """
    Uses the NMS algorithm to filter overlapping Pois.
    This implementation is not complete and is not working correctly in some cases.
    """

    def apply(self, pois: [Poi]) -> [Poi]:
        # return an empty list, if no boxes given
        pois = np.array(pois)
        if len(pois) == 0:
            return []
        treshold = 0.7
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        for poi in pois:
            x1 = x1 + [poi.shape.x]
            y1 = y1 + [poi.shape.y]
            x2 = x2 + [poi.shape.x + poi.shape.w]
            y2 = y2 + [poi.shape.y + poi.shape.h]
        x1 = np.array(x1)
        y1 = np.array(y1)
        x2 = np.array(x2)
        y2 = np.array(y2)
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # We have a least a box of one pixel, therefore the +1
        indices = np.arange(len(x1))
        for i, poi in enumerate(pois):
            temp_indices = indices[indices != i]
            pois_x1 = []
            pois_y1 = []
            pois_x2 = []
            pois_y2 = []
            for j in temp_indices:
                pois_x1 = pois_x1 + [pois[j].shape.x]
                pois_y1 = pois_y1 + [pois[j].shape.y]
                pois_x2 = pois_x2 + [pois[j].shape.x + pois[j].shape.w]
                pois_y2 = pois_y2 + [pois[j].shape.y + pois[j].shape.h]
            xx1 = np.maximum(poi.shape.x, pois_x1)
            yy1 = np.maximum(poi.shape.y, pois_y1)
            xx2 = np.minimum(poi.shape.x + poi.shape.w, pois_x2)
            yy2 = np.minimum(poi.shape.y + poi.shape.h, pois_y2)
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # compute the ratio of overlap
            overlap = (w * h) / areas[temp_indices]
            if np.any(overlap) > treshold:
                indices = indices[indices != i]
        return pois[indices]

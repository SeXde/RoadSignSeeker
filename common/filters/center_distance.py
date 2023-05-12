from common.multifilter import MultiFilter
from common.poi import Poi
import math


class CenterDistance(MultiFilter):
    """
    Unused and incomplete implementation of a filter by center distance.
    """
    def __init__(self, max_distance: float, image_size: (int, int)):
        self.max_distance = max_distance

        self.width = image_size[0]
        self.height = image_size[1]

    def apply(self, pois: [Poi]) -> [Poi]:
        centers = []
        valid_pois = []

        for poi in pois:
            point_x = poi.shape.w / 2 + poi.shape.x
            point_y = poi.shape.h / 2 + poi.shape.y

            centers.append((point_x, point_y))

        for a_index in range(len(centers)):
            for b_index in range(a_index, len(centers)):
                difference = self._calculate_distance(centers[a_index], centers[b_index])

                proportional_x = difference[0] / self.width
                proportional_y = difference[1] / self.height

                distance = math.sqrt(proportional_x * proportional_x + proportional_y * proportional_y)

                if distance > self.max_distance:
                    # TODO: Handle classes (?)
                    valid_pois.append(pois[a_index])

        return []

    @staticmethod
    def _calculate_distance(a: (int, int), b: (int, int)) -> (int, int):
        x = b[0] - a[0]
        y = b[1] - a[1]

        return x, y

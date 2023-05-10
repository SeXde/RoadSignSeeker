from common.multifilter import MultiFilter
from common.poi import Poi
from common.shape import Shape


class AreaScoreBasedFilter(MultiFilter):
    """
    A better version of the AABBFilter.
    """

    @staticmethod
    def _aabb(a: Shape, b: Shape):
        return a.x <= (b.x + b.w) and (a.x + a.w) >= b.x and a.y <= (b.y + b.h) and (a.y + a.h) >= b.y

    @staticmethod
    def _calculate_area(poi: Poi):
        return poi.shape.w * poi.shape.h

    def apply(self, pois: [Poi]) -> [Poi]:
        pois.sort(key=self._calculate_area, reverse=True)
        classified = [None for _ in range(len(pois))]
        clas = 0
        for i in range(len(pois)):
            clas = clas + 1
            if classified[i] is not None:
                continue
            for j in range(len(pois)):
                if i == j:
                    continue
                if self._aabb(pois[i].shape, pois[j].shape):
                    classified[i] = [clas, pois[i]]
                    classified[j] = [clas, pois[j]]
            if classified[i] is None:
                classified[i] = [clas, pois[i]]

        dictionary = {}
        new_pois = []
        for clas, poi in classified:
            if clas in dictionary:
                dictionary[clas] = dictionary[clas] + [poi]
            else:
                dictionary[clas] = [poi]

        for key, pois in dictionary.items():
            new_pois = new_pois + [max(pois, key=lambda desired_poi: desired_poi.score)]
        return new_pois

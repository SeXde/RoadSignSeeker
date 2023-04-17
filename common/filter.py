from common.poi import Poi


class Filter:
    def apply(self, poi: Poi) -> bool:
        return True


def apply_filters(poi: Poi, filters: [Filter]) -> bool:
    for fil in filters:
        if not fil.apply(poi):
            return False
    
    return True

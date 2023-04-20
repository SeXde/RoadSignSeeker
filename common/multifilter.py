from common.poi import Poi


class MultiFilter:
    def apply(self, pois: [Poi]) -> [Poi]:
        return []


def apply_multifilters(pois: [Poi], multifilters: [MultiFilter]) -> [Poi]:
    valid_pois = pois
    for fil in multifilters:
        valid_pois = fil.apply(valid_pois)

    return valid_pois

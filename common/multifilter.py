from common.poi import Poi


class MultiFilter:
    def apply(self, pois: [Poi]) -> bool:
        return True


def apply_multifilters(pois: [Poi], multifilters: [MultiFilter]) -> bool:
    for fil in multifilters:
        if not fil.apply(pois):
            return False

    return True

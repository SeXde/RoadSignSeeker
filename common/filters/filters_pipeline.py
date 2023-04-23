from common.poi import Poi


def apply_filters_pipeline(pois: Poi, func: callable) -> [Poi]:
    return func(pois)

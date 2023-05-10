from common.poi import Poi


class MultiFilter:
    """
    Multifilters are used to filter an array of Poi by comparing them between each others.
    The difference between Multifilter and Filter is that Filters are made to filter ONE Poi,
    while MultiFilters are made to filter a bunch of Pois between each others (removing overlaps, etc)


    Intended usage:
        apply_multifilters(pois, [
            # Subclasses of MultiFilter to filter the pois with.
        ])
    """
    def apply(self, pois: [Poi]) -> [Poi]:
        return []


def apply_multifilters(pois: [Poi], multifilters: [MultiFilter]) -> [Poi]:
    """
    Convenience function to filter all the Poi given using the given MultiFilters
    """
    valid_pois = pois
    for fil in multifilters:
        valid_pois = fil.apply(valid_pois)

    return valid_pois

from common.poi import Poi


class Filter:
    """
    This is the parent class for all single filters.
    Filters are made to process a Poi and return true if it passes a single filter.
    Filters provide a convenient way of handling atomic filters on a Poi.

    Intended usage:
        apply_filters(poi, [
            # Subclasses of filters to filter the poi with.
        ])
    """
    def apply(self, poi: Poi) -> bool:
        """
        This is the method to override in subclasses.
        Return true if the Poi passes the filter.
        """
        return True


def apply_filters(poi: Poi, filters: [Filter]) -> bool:
    """
    Convenience function to apply an array of filters to a Poi
    Returns true when the Poi passes all the filters.
    Stops filtering as soon as one filter is not valid.
    """
    for fil in filters:
        if not fil.apply(poi):
            return False
    
    return True

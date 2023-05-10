import cv2

from common.poi import Poi
from common.shape import Shape


class Detector:
    """
    Parent class for all detectors
    Override the detect method in subclasses to handle all the necessary logic for the detector
    """
    def detect(self, img_path: str) -> [Poi]:
        """
        The detect method will be used in subclasses.
        This method is reserved to detect all the pois (even invalid ones) of an image
        and then return them as instances of Poi
        """
        pass

    @staticmethod
    def _region_to_poi(region, i_rgb, img_path, extra_space=0) -> Poi:
        """
        Process a region of an image and transforms it into a Poi instance.
        Optionally, accepts extra space around the region to
        """
        x, y, w, h = cv2.boundingRect(region)

        shape = Shape(x=max(x - extra_space, 0),
                      y=max(y - extra_space, 0),
                      width=min(w + extra_space * 2, i_rgb.shape[0]),
                      height=min(h + extra_space * 2, i_rgb.shape[1]))

        return Poi(shape, i_rgb, img_path)

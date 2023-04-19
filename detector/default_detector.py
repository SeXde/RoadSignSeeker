import cv2
from common.poi import Poi
from common.shape import Shape


class DefaultDetector:

    def detect(self, img_path: str) -> [Poi]:
        img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
        i_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        i_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        i_gray = cv2.equalizeHist(i_gray)

        mser = cv2.MSER_create(delta=5, max_variation=0.5, max_area=20000)
        regions = mser.detectRegions(i_gray)
        return list(map(lambda r: self._region_to_poi(r, i_rgb), regions))

    @staticmethod
    def _region_to_poi(region, i_rgb) -> Poi:
        x, y, w, h = cv2.boundingRect(region)
        return Poi(Shape(x, y, w, h), i_rgb)

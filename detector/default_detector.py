import cv2
from common.poi import Poi
from common.shape import Shape


class DefaultDetector:

    def detect(self, img_path: str) -> [Poi]:
        pois = []

        img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
        i_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        i_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        i_gray = cv2.equalizeHist(i_gray)

        mser = cv2.MSER_create(delta=5, max_variation=0.5, max_area=20000)
        regions = mser.detectRegions(i_gray)

        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            poi = Poi(Shape(x, y, w, h), i_rgb)
            pois.append(poi)

        return pois

import cv2
from common.poi import Poi
from common.shape import Shape


class Detector:
    def __init__(self, img_path):
        self.pois = self.rectangles(img_path)

    def rectangles(self, img_path: str) -> [Poi]:
        self.pois = []
        img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
        Irgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Igray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        Igray = cv2.equalizeHist(Igray)

        mser = cv2.MSER_create(delta=5, max_variation=0.5, max_area=20000)
        regions = mser.detectRegions(Igray)

        for region in regions:
            x, y, w, h = cv2.boundingRect(region)

            poi = Poi(Shape(x, y, w, h), Irgb)

            self.append(poi)

        return self.pois

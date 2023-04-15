from common.filters.dominant_color_filter import DominantColorFilter
from common.filters.size_filter import SizeFilter
from common.poi import Poi
from common.filters.ratio_filter import RatioFilter
from common.shape import Shape
import cv2
import numpy as np

img = cv2.imread('resources/test_detection/00016.png', cv2.IMREAD_ANYCOLOR)
Irgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
Igray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

Igray = cv2.equalizeHist(Igray)
IMarked = Irgb.copy()

mser = cv2.MSER_create(delta=5, max_variation=0.5, max_area=20000)
polygons = mser.detectRegions(Igray)


def show_image(img, title=''):
    cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


for polygon in polygons[0]:
    x, y, w, h = cv2.boundingRect(polygon)

    poi = Poi(Shape(x, y, w, h), Irgb)

    res = poi.apply_filters([
        SizeFilter(width=(50, None), height=(0, None)),
        RatioFilter(1.5, 4),
        DominantColorFilter(150, 200),
    ])

    #cv2.rectangle(IMarked, (x, y), (x + w, y + h), IMarked.shape, 2)
    if res:
        cv2.rectangle(IMarked, (x, y), (x + w, y + h), IMarked.shape, 2)


show_image(IMarked)
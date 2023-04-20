from common.filter import apply_filters
from common.filters.dominant_color_filter import DominantColorFilter
from common.filters.size_filter import SizeFilter
from common.filters.correlation_filter import CorrelationFilter
from common.filters.aabb_filter import AabbMultiFilter
from common.poi import Poi
from common.filters.ratio_filter import RatioFilter
from common.multifilter import apply_multifilters
from common.shape import Shape
import cv2
import numpy as np

img_path = 'resources/test_detection/00016.png'
img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
Irgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
Igray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

Igray = cv2.equalizeHist(Igray)
IMarked = Irgb.copy()
mask_path = "resources/RoadSign_Mask.png"
img_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mser = cv2.MSER_create(delta=5, max_variation=0.5, max_area=20000)
polygons = mser.detectRegions(Igray)


def show_image(img, title=''):
    cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

pois = []
for polygon in polygons[0]:
    x, y, w, h = cv2.boundingRect(polygon)

    poi = Poi(Shape(x, y, w, h), Irgb, img_path)

    res = apply_filters(poi, [
        SizeFilter(width=(50, None), height=(0, None)),
        RatioFilter(1.5, 4),
        DominantColorFilter(hue=(150, 200), value=(150, None)),
        CorrelationFilter(0.5, img_mask, [150, 150, 50], [250, 255, 255])
    ])

    #cv2.rectangle(IMarked, (x, y), (x + w, y + h), IMarked.shape, 2)
    if res:
        pois.append(poi)
        #cv2.rectangle(IMarked, (x, y), (x + w, y + h), IMarked.shape, 2)

valid_pois = apply_multifilters(pois, [AabbMultiFilter()])
for image in valid_pois:
    show_image(image.get_rgb_rect())
    cv2.rectangle(IMarked, (image.shape.x, image.shape.y), (image.shape.x + image.shape.w, image.shape.y + image.shape.h), (255, 0, 0), 2)
    #cv2.rectangle(IMarked, (image.shape.x, image.shape.y), (image.shape.x + image.shape.w, image.shape.y + image.shape.h), IMarked.shape, 2)
show_image(IMarked)
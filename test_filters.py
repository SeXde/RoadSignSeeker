import numpy as np

from common.poi import Poi
from common.shape import Shape
from common.filters.filters_pipeline import apply_filters_pipeline
import cv2

from pipelines.default_pipeline import default_pipeline
from roadSeekerIo.utils import write_rectangles

img_path = 'resources/train_detection/00002.png'
img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
Irgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
Igray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
Ihsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

Igray = cv2.equalizeHist(Igray)


IMarked = Irgb.copy()
mask_path = "resources/RoadSign_Mask.png"
img_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mser = cv2.MSER_create(delta=5, max_variation=0.5, max_area=99000)
polygons = mser.detectRegions(Igray)


def show_image(img, title=''):
    cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

#a = cv2.inRange(Ihsv, np.array([200 / 2, 200, 200]), np.array([250 / 2, 255, 255]))
#show_image(a)

pois = []
for polygon in polygons[0]:
    x, y, w, h = cv2.boundingRect(polygon)
    pois = pois + [Poi(Shape(x, y, w, h), Irgb, img_path)]


pois = apply_filters_pipeline(pois, default_pipeline)

for poi in pois:
    write_rectangles(IMarked, poi, poi.c)

show_image(IMarked)

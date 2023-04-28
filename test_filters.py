import numpy as np

from common.debug_image import debug_image
from common.filters.dominant_color_filter import DominantColorFilter
from common.poi import Poi
from common.shape import Shape
import cv2

from detector.contour_detector import ContourDetector
from detector.default_detector import DefaultDetector
from pipelines.default_pipeline import DefaultPipeline
from roadSeekerIo.utils import write_rectangles

img_path = 'resources/test_detection/00004.png'
img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
Irgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
Igray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
Ihsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

Igray = cv2.equalizeHist(Igray)


IMarked = Irgb.copy()
mask_path = "resources/RoadSign_Mask.png"
img_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

dominant_color = DominantColorFilter.get_dominant_color(Ihsv)
print(dominant_color)

alpha = (dominant_color[2] / 255 - 0.5) * 50
print(alpha)

lower_blue = (100, 200, 50)
high_blue = (150, 255, 255)

mask = cv2.inRange(Ihsv, lower_blue, high_blue)
debug_image(mask)

pois = ContourDetector(lower_blue, high_blue).detect(img_path)
pois = DefaultPipeline().apply(pois)

for poi in pois:
    write_rectangles(IMarked, poi, poi.c)

debug_image(IMarked)

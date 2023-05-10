import numpy as np

from common.debug_image import debug_image
from common.filters.dominant_color_filter import DominantColorFilter
from common.poi import Poi
from common.shape import Shape
import cv2

from detector.contour_detector import ContourDetector
from detector.default_detector import DefaultDetector
from detector.white_contour_detector import WhiteContourDetector
from pipelines.default_pipeline import DefaultPipeline
from helpers.utils import write_rectangles

img_path = 'resources/test_detection/00010.png'
img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
Irgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
Igray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
Ihsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

Igray = cv2.equalizeHist(Igray)

IMarked = Irgb.copy()

lower_blue = (100, 150, 0)
high_blue = (150, 255, 255)


alpha = 3 # Contrast control (1.0-3.0)
beta = 0 # Brightness control (0-100)

adjusted = cv2.convertScaleAbs(Irgb, alpha=alpha, beta=beta)
adjusted_hsv = cv2.cvtColor(adjusted, cv2.COLOR_RGB2HSV)
mask = cv2.inRange(adjusted_hsv, (0, 250, 250), (180, 255, 255))


pois = ContourDetector(lower_blue, high_blue).detect(img_path)
pois = DefaultPipeline().apply(pois)

for poi in pois:
    write_rectangles(IMarked, poi, poi.c)

debug_image(IMarked)

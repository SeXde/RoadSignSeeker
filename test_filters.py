from common.filter import apply_filters
from common.filters.dominant_color_filter import DominantColorFilter
from common.filters.size_filter import SizeFilter
from common.filters.correlation_filter import CorrelationFilter
from common.filters.aabb_filter import AabbMultiFilter
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

pois = []
for polygon in polygons[0]:
    x, y, w, h = cv2.boundingRect(polygon)

    poi = Poi(Shape(x, y, w, h), Irgb, img_path)

pois = apply_filters_pipeline(pois, default_pipeline)

for poi in pois:
    write_rectangles(IMarked, poi)

show_image(IMarked)
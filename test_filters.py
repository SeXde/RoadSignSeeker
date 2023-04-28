from common.poi import Poi
from common.shape import Shape
import cv2

from detector.contour_detector import ContourDetector
from detector.default_detector import DefaultDetector
from pipelines.default_pipeline import DefaultPipeline
from roadSeekerIo.utils import write_rectangles

img_path = 'resources/train_detection/00003.png'
img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
Irgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
Igray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
Ihsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

Igray = cv2.equalizeHist(Igray)


IMarked = Irgb.copy()
mask_path = "resources/RoadSign_Mask.png"
img_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

def show_image(img, title=''):
    cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


pois = ContourDetector().detect(img_path)
pois = DefaultPipeline().apply(pois)

print(len(pois))

for poi in pois:
    write_rectangles(IMarked, poi, poi.c)

show_image(IMarked)

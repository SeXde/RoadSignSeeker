import cv2

from common.filters.aabb_filter import AabbMultiFilter
from common.poi import Poi
from common.shape import Shape

img_path = 'resources/train_detection/00016.png'
img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)

poi1 = Poi(Shape(0, 0, 5000, 5000), img, img_path)
poi1.score = 1

poi2 = Poi(Shape(100, 100, 4, 4), img, img_path)
poi2.score = 0

pois = [
    poi1,
    poi2
]

final_pois = AabbMultiFilter().apply(pois)

print('LEN', len(final_pois), final_pois[0].shape.x)

print("RES", AabbMultiFilter.aabb_test(Shape(172, 241, 173, 147), Shape(193, 266, 128, 42)))


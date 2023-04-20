from common.multifilter import MultiFilter
from common.poi import Poi
import numpy as np
from common.shape import Shape


class AabbMultiFilter(MultiFilter):

    @staticmethod
    def _aabb_test(a: Shape, b: Shape) -> bool:
        return a.x <= (b.x + b.w) and (a.x + a.w) >= b.x and a.y <= (b.y + b.h) and (a.y + a.h) >= b.y

    def apply(self, pois: [Poi]) -> [Poi]:
        num_images = len(pois)

        classes = np.zeros(num_images)
        c = 1

        classes[0] = c

        for a_index in range(num_images):
            shape_a = pois[a_index]

            for b_index in range(a_index, num_images):
                shape_b = pois[b_index]

                if a_index == b_index:
                    continue

                if self._aabb_test(shape_a.shape, shape_b.shape):
                    classes[b_index] = classes[a_index]
                else:
                    c = c + 1
                    classes[b_index] = c
                    break

        available_classes = np.unique(classes)
        print(classes, available_classes)
        selected = []
        last_class = 1
        for c_index in range(len(available_classes)):
            clas = available_classes[c_index]
            for a_index in range(num_images):
                img_class = classes[a_index]
                if clas == img_class:
                    rect = pois[a_index]
                    selected.append(rect)
                    break
        return selected

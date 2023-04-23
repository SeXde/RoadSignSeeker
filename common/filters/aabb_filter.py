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

        if num_images == 0:
            return []

        pois = sorted(pois, reverse=True, key=lambda poi: poi.shape.w * poi.shape.h)

        classes = np.zeros(num_images)

        c = 1
        classes[0] = c

        for a_index in range(num_images):
            shape_a = pois[a_index].shape

            if classes[a_index] != 0:
                pois[a_index].c = classes[a_index]
                continue

            for b_index in range(num_images):
                shape_b = pois[b_index].shape

                if self._aabb_test(shape_a, shape_b):
                    classes[a_index] = classes[b_index]
                else:
                    c = c + 1

            if classes[a_index] == 0:
                c = c + 1
                classes[a_index] = c

            pois[a_index].c = classes[a_index]

        available_classes = np.unique(classes)
        print(classes)

        # 34, 7882

        # (clas, index)
        class_score = dict()

        for clas in available_classes:
            class_score[clas] = None

            for a_index in range(num_images):
                img_class = classes[a_index]

                if clas == img_class and (class_score[clas] is None or pois[a_index].score > class_score[clas][0]):
                    class_score[clas] = (pois[a_index].score, a_index)

        selected = []
        for clas in available_classes:
            _, index = class_score[clas]
            selected.append(pois[int(index)])

        return selected

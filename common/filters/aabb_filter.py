import cv2

from common.multifilter import MultiFilter
from common.poi import Poi
import numpy as np
from common.shape import Shape
from roadSeekerIo.utils import draw_classes, generate_random_colors, hsv_to_rgb


class AabbMultiFilter(MultiFilter):

    @staticmethod
    def _aabb_test(a: Shape, b: Shape) -> bool:
        return a.x <= (b.x + b.w) and (a.x + a.w) >= b.x and a.y <= (b.y + b.h) and (a.y + a.h) >= b.y

    def apply(self, pois: [Poi]) -> [Poi]:
        num_images = len(pois)

        if num_images == 0:
            return []

        # return self._classify_images(self._aabb_test, pois)

        pois = sorted(pois, reverse=True, key=lambda poi: poi.shape.w * poi.shape.h)

        classes = np.zeros(num_images)

        c = 0
        classes[0] = c

        for a_index in range(num_images):
            shape_a = pois[a_index].shape
            c = c + 1

            if classes[a_index] != 0:
                pois[a_index].c = classes[a_index]
                continue

            for b_index in range(num_images):
                shape_b = pois[b_index].shape

                if self._aabb_test(shape_a, shape_b):
                    classes[a_index] = classes[b_index]

            if classes[a_index] == 0:
                c = c + 1
                classes[a_index] = c

            pois[a_index].c = classes[a_index]

        available_classes = np.unique(classes)
        draw_classes(pois, classes)

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

    def _classify_images(self, match_function: callable, pois: [Poi]) -> [Poi]:
        pois = sorted(pois, reverse=False, key=lambda poi: poi.shape.w * poi.shape.h)

        total_pois = len(pois)

        all_children = {}

        for this_index in range(total_pois):
            this_shape = pois[this_index].shape

            for other_index in range(this_index + 1, total_pois):
                other_shape = pois[other_index].shape

                if match_function(this_shape, other_shape):
                    other_children = all_children[other_index] if other_index in all_children else []

                    all_children[other_index] = other_children + [this_index]

        selected = []

        colors = generate_random_colors(len(all_children.keys()))

        i = 0

        for parent, children in all_children.items():
            parent_poi = self._get_pois(pois, [parent])
            children_pois = self._get_pois(pois, children)

            if len(children) == 1:
                selected += parent_poi

            parent_area = self._find_area(parent_poi)
            children_area = self._find_area(children_pois)

            diff = children_area / parent_area

            for child in children_pois:
                color = hsv_to_rgb(colors[i] / len(colors), 1, 1)
                child.c = (color[0] * 255, color[1] * 255, color[2] * 255)

            i += 1

            if len(children) == 3 and 0.9 < diff < 1.5:
                selected += children_pois

            if 0.9 < diff < 1.1:
                selected += children_pois

                parent_poi[0].c = (0, 0, 0)
                #return children_pois #+ parent_poi

        return selected

    def _find_area(self, pois: [Poi]) -> float:
        area = 0
        for poi in pois:
            area += poi.shape.w * poi.shape.h

        return area

    def _get_pois(self, all_pois: [Poi], find: [int]) -> [Poi]:
        return [all_pois[child] for child in find]
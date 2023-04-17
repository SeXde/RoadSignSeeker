import cv2
from pathlib import Path

from common.poi import Poi
from roadSeekerIo.panel import Panel
from roadSeekerIo.paths import GENERATED_IMG_PATH


def save_panels(img_path: str, panels: [Panel]):
    image = cv2.imread(img_path)
    for panel in panels:
        cv2.rectangle(image, panel.upper_edge, panel.lower_edge, (0, 0, 255), 2)
        (x, y) = panel.lower_edge
        cv2.putText(image, str(panel.score), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 233, 255), 1)
    cv2.imwrite("{}/{}".format(GENERATED_IMG_PATH, Path(img_path).name), image)


def poi_to_panel(poi: Poi, file_path: str) -> Panel:
    # TODO compute score
    return Panel(file_path=file_path, upper_edge=(poi.shape.x, poi.shape.y + poi.shape.h),
                 lower_edge=(poi.shape.x + poi.shape.w, poi.shape.y), score=1)

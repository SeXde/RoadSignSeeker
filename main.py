import argparse
import os
import shutil

from detector.contour_detector import ContourDetector
from detector.default_detector import DefaultDetector
from pipelines.nms_pipeline import NMSPipeline
from helpers.paths import GENERATED_IMG_PATH
from pipelines.default_pipeline import DefaultPipeline
from helpers.utils import save_panels
from helpers.colors import *
from pipelines.no_overlap_pipeline import NoOverlapPipeline

DETECTORS = {
    'default': DefaultDetector(),
    'contour-blue': ContourDetector(LOW_BLUE, HIGH_BLUE),
    'contour-white': ContourDetector(LOW_WHITE, HIGH_WHITE)
}

FILTER_PIPELINES = {
    'default': DefaultPipeline(),
    'nms': NMSPipeline(),
    'no-overlap': NoOverlapPipeline(),
}


def validate_and_build_args(program_args):
    chosen_detector = DETECTORS.get(program_args.detector)

    if chosen_detector is None:
        raise ValueError("{} is not a valid detector".format(program_args.detector))

    chosen_filter_pipeline = FILTER_PIPELINES.get(program_args.filter_pipeline)

    if chosen_filter_pipeline is None:
        raise ValueError("{} is not a valid filter pipeline".format(program_args.filter_pipeline))

    if not os.path.exists(program_args.train_path):
        raise ValueError("Train path {} does not exist".format(program_args.train_path))

    if not os.path.exists(program_args.test_path):
        raise ValueError("Test path {} does not exist".format(program_args.test_path))

    return chosen_detector, chosen_filter_pipeline


def detect_and_write_panels(image_path: str):
    print(f"Processing {image_path}...")

    pois = detector.detect(image_path)
    filtered_pois = filter_pipeline.apply(pois)

    save_panels(image_path, filtered_pois)

    for poi in filtered_pois:
        poi.save_to_file()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains and executes a given detector over a set of testing images')
    parser.add_argument(
        '--detector', type=str, nargs=1, default="contour-blue", help='Detector implementation: {}'.format(list(DETECTORS.keys())))
    parser.add_argument(
        '--filter_pipeline', type=str, nargs=1, default="no-overlap",
        help='Filter pipeline implementation: {}'.format(list(FILTER_PIPELINES.keys())))
    parser.add_argument(
        '--train_path', default="resources/train_detection", help='Select the training data dir')
    parser.add_argument(
        '--test_path', default="resources/test_detection", help='Select the testing data dir')

    args = parser.parse_args()
    detector, filter_pipeline = validate_and_build_args(args)

    if os.path.exists(GENERATED_IMG_PATH):
        shutil.rmtree(GENERATED_IMG_PATH)

    os.makedirs(GENERATED_IMG_PATH)

    image_paths = os.listdir(args.test_path)
    image_paths = list(filter(lambda path: 'png' in path or 'PNG' in path, image_paths))

    for image_path in image_paths:
        detect_and_write_panels(args.test_path + "/" + image_path)

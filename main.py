import argparse
import os
import cv2

from detector.default_detector import DefaultDetector
from pipelines.default_pipeline import default_pipeline
from roadSeekerIo.utils import save_panels

DETECTORS = {
    'default': DefaultDetector(),
    'improved': "return our_impl"  # TODO
}

FILTER_PIPELINES = {
    'default': default_pipeline,  # TODO
    'improved': "return improved pipeline impl"  # TODO
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
    pois = detector.detect(image_path)
    filtered_pois = filter_pipeline(pois)
    print(image_path, len(filtered_pois))
    save_panels(image_path, filtered_pois)
    for poi in filtered_pois:
        poi.save_to_file()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains and executes a given detector over a set of testing images')
    parser.add_argument(
        '--detector', type=str, nargs=1, default="default", help='Detector implementation: {}'.format(list(DETECTORS.keys())))
    parser.add_argument(
        '--filter_pipeline', type=str, nargs=1, default="default",
        help='Filter pipeline implementation: {}'.format(list(FILTER_PIPELINES.keys())))
    parser.add_argument(
        '--train_path', default="resources/train_detection", help='Select the training data dir')
    parser.add_argument(
        '--test_path', default="resources/test_detection", help='Select the testing data dir')

    args = parser.parse_args()
    detector, filter_pipeline = validate_and_build_args(args)

    image_paths = os.listdir(args.test_path)
    image_paths = list(filter(lambda path: 'png' in path or 'PNG' in path, image_paths))
    for image_path in image_paths:
        detect_and_write_panels(args.test_path + "/" + image_path)

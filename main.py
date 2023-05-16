import argparse
import os
import shutil

import cv2

from classifier.bayes_classifier import BayesClassifier
from classifier.knn_classifier import KnnClassifier
from classifier.naive_bayes_classifier import NaiveBayesClassifier
from common.debug_image import debug_image
from detector.contour_detector import ContourDetector
from detector.default_detector import DefaultDetector
from dimension.lda_dim_reduction import LdaDimReduction
from dimension.pca_dim_reduction import PcaDimReduction
from ocr.panel_ocr import PanelTextOcr
from pipelines.nms_pipeline import NMSPipeline
from helpers.paths import GENERATED_IMG_PATH, classes, OCR_PATH
from pipelines.default_pipeline import DefaultPipeline
from helpers.utils import save_panels, read_panels_and_build_classes
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

CLASSIFIERS = {
    'bayes': BayesClassifier(),
    'knn': KnnClassifier(),
    'naive': NaiveBayesClassifier()
}

DIMENSION = {
    'lda': LdaDimReduction(),
    'pca': PcaDimReduction()
}


def validate_and_build_args(program_args):
    chosen_detector = DETECTORS.get(str(program_args.detector))
    if chosen_detector is None:
        raise ValueError("{} is not a valid detector".format(program_args.detector))

    chosen_filter_pipeline = FILTER_PIPELINES.get(str(program_args.filter_pipeline))
    if chosen_filter_pipeline is None:
        raise ValueError("{} is not a valid filter pipeline".format(program_args.filter_pipeline))

    if not os.path.exists(program_args.train_path):
        raise ValueError("Train path {} does not exist".format(program_args.train_path))

    if not os.path.exists(program_args.test_path):
        raise ValueError("Test path {} does not exist".format(program_args.test_path))

    chosen_classifier = CLASSIFIERS.get(str(program_args.classifier))

    if chosen_classifier is None:
        raise ValueError("{} is not a valid classifier".format(program_args.classifier))

    chosen_dimension = DIMENSION.get(str(program_args.dimension))

    if chosen_dimension is None:
        raise ValueError("{} is not a valid dimension reduction system".format(program_args.dimension))

    if not os.path.exists(program_args.train_path_ocr):
        raise ValueError("Train path ocr {} does not exist".format(program_args.train_path))

    return chosen_detector, chosen_filter_pipeline, chosen_classifier, chosen_dimension


def detect_and_write_panels(image_path: str, panel_ocr_desired: PanelTextOcr):
    print(f"Processing {image_path}...")

    pois = detector.detect(image_path)
    filtered_pois = filter_pipeline.apply(pois)

    image_bgr = save_panels(image_path, filtered_pois)
    if args.show_image:
        debug_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    for poi in filtered_pois:
        panel_ocr_desired.classify_panel(cv2.imread(image_path), (poi.shape.y, poi.shape.x, poi.shape.h, poi.shape.w),
                                         os.path.basename(image_path),
                                         show_image=args.show_image, output_file=ocr_path_result)
        poi.save_to_file()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Trains and executes a given detector over a set of testing images')
    parser.add_argument(
        '--detector', type=str, default="contour-blue",
        help='Detector implementation: {}'.format(list(DETECTORS.keys())))
    parser.add_argument(
        '--filter_pipeline', type=str, default="no-overlap",
        help='Filter pipeline implementation: {}'.format(list(FILTER_PIPELINES.keys())))
    parser.add_argument(
        '--train_path', default="resources/train_detection", help='Select the training data dir')
    parser.add_argument(
        '--test_path', default="resources/test_detection", help='Select the testing data dir')
    parser.add_argument('--classifier', type=str, default="bayes",
                        help='List of implemented classifiers: {}'.format(list(CLASSIFIERS.keys())))
    parser.add_argument('--dimension', type=str, default="lda",
                        help='List of implemented dimension reduction systems: {}'.format(list(DIMENSION.keys())))
    parser.add_argument(
        '--train_path_ocr', default="resources/train_ocr", help='Select the training data dir')
    parser.add_argument(
        '--show_image', type=bool, default=False,
        help='True if you want yo display detected panels and text inside them. Use "n" '
             'to skip images')

    args = parser.parse_args()
    detector, filter_pipeline, classifier, dimension = validate_and_build_args(args)
    ocr_path_result = OCR_PATH + "/resultado.txt"
    if os.path.exists(ocr_path_result):
        os.remove(ocr_path_result)
    if os.path.exists(GENERATED_IMG_PATH):
        shutil.rmtree(GENERATED_IMG_PATH)

    os.makedirs(GENERATED_IMG_PATH)

    panel_ocr = PanelTextOcr(classifier, dimension)
    letters = []
    train_images_bgr = []

    print("Loading training images ...")
    for class_path, class_letter in classes.items():
        class_path = args.train_path_ocr + class_path
        class_path_image_names = os.listdir(class_path)
        train_images_bgr = train_images_bgr + list(
            map(lambda p: read_panels_and_build_classes(class_path, p, class_letter, letters),
                filter(lambda p: 'png' in p or 'PNG' in p, class_path_image_names))
        )

    print("Training ocr with {} images ...".format(len(train_images_bgr)))
    panel_ocr.train_ocr(train_images_bgr, letters)
    print("Ocr trained!")

    image_paths = os.listdir(args.test_path)
    image_paths = list(filter(lambda path: 'png' in path or 'PNG' in path, image_paths))

    for image_path in image_paths:
        detect_and_write_panels(args.test_path + "/" + image_path, panel_ocr)

import argparse
import os

import cv2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import ransac.line_detector
from classifier.bayes_classifier import BayesClassifier
from classifier.knn_classifier import KnnClassifier
from classifier.naive_bayes_classifier import NaiveBayesClassifier
from common.debug_image import debug_image
from dimension.lda_dim_reduction import LdaDimReduction
from dimension.pca_dim_reduction import PcaDimReduction
from featureExtractor.feature_extractor import FeatureExtractor
from ransac import line_detector
from roadSeekerIo.paths import classes, OCR_PATH
from threshold.default_threshold import DefaultThreshold

CLASSIFIERS = {
    'default': BayesClassifier(),
    'knn': KnnClassifier(),
    'naive': NaiveBayesClassifier()
}

DIMENSION = {
    'default': LdaDimReduction(),
    'pca': PcaDimReduction()
}


def validate_and_build_args(program_args):
    chosen_classifier = CLASSIFIERS.get(program_args.classifier)

    if chosen_classifier is None:
        raise ValueError("{} is not a valid classifier".format(program_args.classifier))

    chosen_dimension = DIMENSION.get(program_args.dimension)

    if chosen_dimension is None:
        raise ValueError("{} is not a valid dimension reduction system".format(program_args.dimension))

    if not os.path.exists(program_args.train_path):
        raise ValueError("Train path {} does not exist".format(program_args.train_path))

    if not os.path.exists(program_args.panels_path):
        raise ValueError("Panels path {} does not exist".format(program_args.panels_path))

    return chosen_classifier, chosen_dimension


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes a given classifier for OCR over testing images')
    parser.add_argument('--classifier', type=str, nargs=1, default="default",
                        help='List of implemented classifiers: {}'.format(list(CLASSIFIERS.keys())))
    parser.add_argument('--dimension', type=str, nargs=1, default="default",
                        help='List of implemented dimension reduction systems: {}'.format(list(DIMENSION.keys())))
    parser.add_argument(
        '--train_path', default="resources/train_ocr", help='Select the training data dir')
    parser.add_argument(
        '--panels_path', default="resources/test_ocr_panels", help='Select the panels data dir')

    args = parser.parse_args()
    # todo check args
    # Train classifier
    default_threshold = DefaultThreshold()
    feature_extractor = FeatureExtractor()
    classifier, dimension = validate_and_build_args(args)
    ocr_path_result = OCR_PATH + "/resultado.txt"
    if os.path.exists(ocr_path_result):
        os.remove(ocr_path_result)
    c = []
    e = []
    for path, letter in classes.items():
        image_paths = os.listdir(args.train_path + path)
        image_paths = list(filter(lambda p: 'png' in p or 'PNG' in p, image_paths))
        print("loading path: {}{}".format(args.train_path, path))
        for image_path in image_paths:
            image = cv2.imread("{}{}/{}".format(args.train_path, path, image_path))
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            contours = default_threshold.threshold_image_train_panel(image_gray)
            feature_extractor.extract(contours, ord(letter), image_gray, c, e)
    dimension.create(c, e)
    cr = dimension.reduce(c)
    classifier.create(cr, e)

    panels = os.listdir(args.panels_path)
    panels = list(filter(lambda p: 'png' in p or 'PNG' in p, panels))
    for panel in panels:
        print("processing panel: {}/{}".format(args.panels_path, panel))
        image = cv2.imread("{}/{}".format(args.panels_path, panel))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = 255 - image_gray
        contours = default_threshold.threshold_image_train_panel(image_gray)
        l_detector = line_detector.LineDetector()
        lines = l_detector.detect_lines(contours, image_rgb)
        text = ""
        for line in lines:
            for letter in line:
                c = []
                feature_extractor.extract([letter], 0, image_gray, c, [], True)
                xr = dimension.reduce(c)
                predicted_letter = classifier.classify(xr)
                text = text + predicted_letter
                x, y, w, h = cv2.boundingRect(letter)
                cv2.putText(image_rgb, predicted_letter, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
            text = text + "+"
        text = text[:-1]
        # debug_image(image_rgb)
        x, y = image_gray.shape
        panel_result = "{};0;0;{};{};1;1;{}\n".format(panel, y - 1, x - 1, text)
        with open(ocr_path_result, 'a+') as f:
            f.write(panel_result)

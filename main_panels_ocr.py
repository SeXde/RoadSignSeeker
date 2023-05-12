import argparse
import os

from classifier.bayes_classifier import BayesClassifier
from classifier.knn_classifier import KnnClassifier
from classifier.naive_bayes_classifier import NaiveBayesClassifier
from dimension.lda_dim_reduction import LdaDimReduction
from dimension.pca_dim_reduction import PcaDimReduction
from helpers.paths import classes, OCR_PATH
from helpers.utils import read_panels_and_build_classes, read_panels
from ocr.panel_ocr import PanelTextOcr

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
    parser.add_argument(
        '--show_image', default=False, help='True if you want yo display detected text inside panel. Use "n" '
                                           'to skip images')

    args = parser.parse_args()
    classifier, dimension = validate_and_build_args(args)
    ocr_path_result = OCR_PATH + "/resultado.txt"
    if os.path.exists(ocr_path_result):
        os.remove(ocr_path_result)
    panel_ocr = PanelTextOcr()
    panel_ocr.create(classifier, dimension)
    letters = []
    train_images_bgr = []

    print("Loading training images ...")
    for class_path, class_letter in classes.items():
        class_path = args.train_path + class_path
        class_path_image_names = os.listdir(class_path)
        train_images_bgr = train_images_bgr + list(
            map(lambda p: read_panels_and_build_classes(class_path, p, class_letter, letters),
                filter(lambda p: 'png' in p or 'PNG' in p, class_path_image_names))
            )

    print("Training ocr with {} images ...".format(len(train_images_bgr)))
    panel_ocr.train_ocr(train_images_bgr, letters)
    print("Ocr trained!")
    print("Loading panel images ...")
    panel_names = list(filter(lambda p: 'png' in p or 'PNG' in p, os.listdir(args.panels_path)))
    panel_images_bgr = list(map(lambda p: read_panels(args.panels_path, p), panel_names))
    print("Successfully loaded {} panel images".format(len(panel_images_bgr)))
    for panel_name, panel_image_bgr in zip(panel_names, panel_images_bgr):
        x, y, _ = panel_image_bgr.shape
        panel_ocr.classify_panel(panel_image_bgr, (0, 0, x - 1, y - 1), panel_name, output_file=ocr_path_result,
                                 show_image=args.show_image)
    print("All panels has been classified!")

import argparse
import os

import cv2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import ransac.line_detector
from classifier.bayes_classifier import BayesClassifier
from common.debug_image import debug_image
from dimension.lda_dim_reduction import LdaDimReduction
from featureExtractor.feature_extractor import FeatureExtractor
from ransac import line_detector
from threshold.default_threshold import DefaultThreshold

classes = {
    '/train_ocr/0': '0',
    '/train_ocr/1': '1',
    '/train_ocr/2': '2',
    '/train_ocr/3': '3',
    '/train_ocr/4': '4',
    '/train_ocr/5': '5',
    '/train_ocr/6': '5',
    '/train_ocr/7': '7',
    '/train_ocr/8': '8',
    '/train_ocr/9': '9',
    '/train_ocr/may/A': 'A',
    '/train_ocr/may/B': 'B',
    '/train_ocr/may/C': 'C',
    '/train_ocr/may/D': 'D',
    '/train_ocr/may/E': 'E',
    '/train_ocr/may/F': 'F',
    '/train_ocr/may/G': 'G',
    '/train_ocr/may/H': 'H',
    '/train_ocr/may/I': 'I',
    '/train_ocr/may/J': 'J',
    '/train_ocr/may/K': 'K',
    '/train_ocr/may/L': 'L',
    '/train_ocr/may/M': 'M',
    '/train_ocr/may/N': 'N',
    '/train_ocr/may/O': 'O',
    '/train_ocr/may/P': 'P',
    '/train_ocr/may/Q': 'Q',
    '/train_ocr/may/R': 'R',
    '/train_ocr/may/S': 'S',
    '/train_ocr/may/T': 'T',
    '/train_ocr/may/U': 'U',
    '/train_ocr/may/V': 'V',
    '/train_ocr/may/W': 'W',
    '/train_ocr/may/X': 'X',
    '/train_ocr/may/Y': 'Y',
    '/train_ocr/may/Z': 'Z',
    '/train_ocr/min/a': 'a',
    '/train_ocr/min/b': 'b',
    '/train_ocr/min/c': 'c',
    '/train_ocr/min/d': 'd',
    '/train_ocr/min/e': 'e',
    '/train_ocr/min/f': 'f',
    '/train_ocr/min/g': 'g',
    '/train_ocr/min/h': 'h',
    '/train_ocr/min/i': 'i',
    '/train_ocr/min/j': 'j',
    '/train_ocr/min/k': 'k',
    '/train_ocr/min/l': 'l',
    '/train_ocr/min/m': 'm',
    '/train_ocr/min/n': 'n',
    '/train_ocr/min/o': 'o',
    '/train_ocr/min/p': 'p',
    '/train_ocr/min/q': 'q',
    '/train_ocr/min/r': 'r',
    '/train_ocr/min/s': 's',
    '/train_ocr/min/t': 't',
    '/train_ocr/min/u': 'u',
    '/train_ocr/min/v': 'v',
    '/train_ocr/min/w': 'w',
    '/train_ocr/min/x': 'x',
    '/train_ocr/min/y': 'y',
    '/train_ocr/min/z': 'z',
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='/trains and executes a given detector over a set of testing images')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="", help='Detector string name')
    parser.add_argument(
        '--train_path', default="resources", help='Select the /training data dir')
    parser.add_argument(
        '--test_path', default="", help='Select the testing data dir')

    args = parser.parse_args()
    # todo check args
    # Train classifier
    default_threshold = DefaultThreshold()
    feature_extractor = FeatureExtractor()
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
    lda_dim_reduction = LdaDimReduction()
    lda_dim_reduction.create(c, e)
    cr = lda_dim_reduction.reduce(c)
    bayes_classifier = BayesClassifier()
    bayes_classifier.create(cr, e)

    ocr_path = "/test_ocr_panels"
    panels = os.listdir(args.train_path + ocr_path)
    panels = list(filter(lambda p: 'png' in p or 'PNG' in p, panels))
    for panel in panels:
        print("processing panel: {}{}/{}".format(args.train_path, ocr_path, panel))
        image = cv2.imread("{}{}/{}".format(args.train_path, ocr_path, panel))
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
                xr = lda_dim_reduction.reduce(c)
                predicted_letter = bayes_classifier.classify(xr)
                text = text + predicted_letter
                x, y, w, h = cv2.boundingRect(letter)
                cv2.putText(image_rgb, predicted_letter, (x,  y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
            text = text + "+"
        text = text[:-1]
        #debug_image(image_rgb)
        print(text)











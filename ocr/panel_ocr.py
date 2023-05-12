import os

import cv2

from classifier.classifier import Classifier
from common.debug_image import debug_image
from dimension.dim_reduction import DimReduction
from featureExtractor.feature_extractor import FeatureExtractor
from ransac.line_detector import LineDetector
from threshold.default_threshold import DefaultThreshold


class PanelTextOcr:

    def __init__(self):
        self.line_detector = LineDetector()
        self.feature_extractor = FeatureExtractor()
        self.default_threshold = DefaultThreshold()
        self.dimension = None
        self.classifier = None

    def create(self, classifier: Classifier, dimension: DimReduction):
        self.classifier = classifier
        self.dimension = dimension

    def train_ocr(self, images_bgr, classes):
        if len(images_bgr) != len(classes):
            raise AttributeError("images and classes len should be the same")
        c = []
        e = []
        for image_BGR, letter in zip(images_bgr, classes):
            image_gray = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2GRAY)
            contours = self.default_threshold.threshold_image_train_panel(image_gray)
            self.feature_extractor.extract(contours, ord(letter), image_gray, c, e)
        self.dimension.create(c, e)
        cr = self.dimension.reduce(c)
        self.classifier.create(cr, e)

    def classify_panel(self, image_bgr, panel_coordinates, image_bgr_name, output_file=None, show_image=False):
        x, y, w, h = panel_coordinates
        panel_bgr = image_bgr[x: x + w, y: y + h]
        panel_rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
        panel_gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY)
        panel_gray = 255 - panel_gray
        contours = self.default_threshold.threshold_image_train_panel(panel_gray)
        lines = self.line_detector.detect_lines(contours, panel_rgb)
        text = ""
        for line in lines:
            for letter in line:
                c = []
                self.feature_extractor.extract([letter], 0, panel_gray, c, [], True)
                xr = self.dimension.reduce(c)
                predicted_letter = self.classifier.classify(xr)
                text = text + predicted_letter
                x, y, w, h = cv2.boundingRect(letter)
                cv2.putText(panel_rgb, predicted_letter, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(panel_rgb, (x, y), (x + w, y + h), (255, 0, 0), 1)
            text = text + "+"
        text = text[:-1]
        x, y = panel_gray.shape
        panel_result = "{};0;0;{};{};1;1;{}\n".format(image_bgr_name, y - 1, x - 1, text)
        if output_file is not None:
            with open(output_file, 'a+') as f:
                f.write(panel_result)
        if show_image:
            debug_image(panel_rgb)

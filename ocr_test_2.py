import os

import cv2

from classifier.bayes_classifier import BayesClassifier
from dimension.lda_dim_reduction import LdaDimReduction
from ocr.panel_ocr import PanelTextOcr
from helpers.paths import classes, OCR_PATH
from helpers.utils import read_panels_and_build_classes, read_panels

classifier = BayesClassifier()
dimension = LdaDimReduction()

panel_ocr = PanelTextOcr(classifier, dimension)

letters = []
train_images_bgr = []

print("Loading training images ...")

for class_path, class_letter in classes.items():
    class_path = "resources/train_ocr" + class_path

    class_path_image_names = os.listdir(class_path)
    train_images_bgr = train_images_bgr + list(
        map(lambda p: read_panels_and_build_classes(class_path, p, class_letter, letters),
            list(filter(lambda p: 'png' in p or 'PNG' in p, class_path_image_names)))
        )

print("Training ocr with {} images ...".format(len(train_images_bgr)))

panel_ocr.train_ocr(train_images_bgr, letters)

print("Ocr trained!")

print("Loading panel images ...")

image = cv2.imread("resources/test_ocr_panels/00003_0.png")
x, y, _ = image.shape

text = panel_ocr.classify_panel(image, (0, 0, x - 1, y - 1), "test", show_image=True)

print(text)

print("All panels has been classified!")

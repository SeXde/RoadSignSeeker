# Asignatura de Visión Artificial (URJC). Script de evaluación.
# @author Jose M. Buenaposada (josemiguel.buenaposada@urjc.es)
# @date 2023


import argparse
import os
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from classifier.bayes_classifier import BayesClassifier
from classifier.knn_classifier import KnnClassifier
from classifier.naive_bayes_classifier import NaiveBayesClassifier
from dimension.lda_dim_reduction import LdaDimReduction
from dimension.pca_dim_reduction import PcaDimReduction
from feature_extractor.feature_extractor import FeatureExtractor
from helpers.paths import classes
from threshold.improved_threshold import ImprovedThreshold

CLASSIFIERS = {
    'bayes': BayesClassifier(),
    'knn': KnnClassifier(),
    'naive': NaiveBayesClassifier()
}

DIMENSION = {
    'lda': LdaDimReduction(),
    'pca': PcaDimReduction()
}

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.get_cmap('Blues')):
    '''
    Given a confusión matrix in cm (np.array) it plots it in a fancy way.
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, range(cm.shape[0]))
    plt.yticks(tick_marks, range(cm.shape[0]))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    ax = plt.gca()
    width = cm.shape[1]
    height = cm.shape[0]

    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[y,x]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')


def validate_and_build_args(program_args):
    chosen_classifier = CLASSIFIERS.get(str(program_args.classifier))

    if chosen_classifier is None:
        raise ValueError("{} is not a valid classifier".format(program_args.classifier))

    chosen_dimension = DIMENSION.get(str(program_args.dimension))

    if chosen_dimension is None:
        raise ValueError("{} is not a valid dimension reduction system".format(program_args.dimension))

    if not os.path.exists(program_args.train_path):
        raise ValueError("Train path {} does not exist".format(program_args.train_path))

    if not os.path.exists(program_args.validation_path):
        raise ValueError("Validation path {} does not exist".format(program_args.validation_path))

    return chosen_classifier, chosen_dimension

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes a given classifier for OCR over testing images')
    parser.add_argument('--classifier', type=str, default="bayes", help='List of implemented classifiers: {}'.format(list(CLASSIFIERS.keys())))
    parser.add_argument('--dimension', type=str, default="lda",
                        help='List of implemented dimension reduction systems: {}'.format(list(DIMENSION.keys())))
    parser.add_argument(
        '--train_path', default="resources/train_ocr", help='Select the training data dir')
    parser.add_argument(
        '--validation_path', default="resources/validation_ocr", help='Select the validation data dir')

    args = parser.parse_args()


    # 1) Cargar las imágenes de entrenamiento y sus etiquetas. 
    # También habrá que extraer los vectores de características asociados (en la parte básica 
    # umbralizar imágenes, pasar findContours y luego redimensionar)
    default_threshold = ImprovedThreshold((2, 2))
    feature_extractor = FeatureExtractor()
    classifier, dimension = validate_and_build_args(args)
    c_train = []
    e_train = []
    time_before = time()
    for path, letter in classes.items():
        image_paths = os.listdir(args.train_path + path)
        image_paths = list(filter(lambda p: 'png' in p or 'PNG' in p, image_paths))
        print("loading path: {}{}".format(args.train_path, path))
        for image_path in image_paths:
            image = cv2.imread("{}{}/{}".format(args.train_path, path, image_path))
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            train_contours = default_threshold.threshold_image(image_gray, True)
            feature_extractor.extract(train_contours, ord(letter), image_gray, c_train, e_train)

    dimension.create(c_train, e_train)
    cr_train = dimension.reduce(c_train)
    classifier.create(cr_train, e_train)

    gt_labels = []
    predicted_labels = []
    for path, letter in classes.items():
        image_paths = os.listdir(args.validation_path + path)
        image_paths = list(filter(lambda p: 'png' in p or 'PNG' in p, image_paths))
        print("loading path: {}{}".format(args.validation_path, path))
        for image_path in image_paths:
            gt_labels.append(letter)
            image = cv2.imread("{}{}/{}".format(args.validation_path, path, image_path))
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            validation_contours = default_threshold.threshold_image(image_gray, True)
            c_validation = []
            feature_extractor.extract(validation_contours, ord(letter), image_gray, c_validation, [])
            c_validation = dimension.reduce(c_validation)
            predicted_labels.append(classifier.classify(c_validation))


    # 5) Evaluar los resultados
    print("Elapsed time = {} seconds".format(round(time() - time_before, 2)))
    accuracy = sklearn.metrics.accuracy_score(gt_labels, predicted_labels)
    print("Accuracy = ", accuracy)
    f1_macro = sklearn.metrics.f1_score(gt_labels, predicted_labels, average='macro')
    print("F1 macro = ", f1_macro)
    confusion_matrix = confusion_matrix(gt_labels, predicted_labels)
    ConfusionMatrixDisplay(confusion_matrix).plot()
    plt.show()



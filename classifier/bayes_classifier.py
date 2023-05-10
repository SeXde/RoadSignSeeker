import cv2
import numpy as np

from classifier.classifier import Classifier


class BayesClassifier(Classifier):

    def __init__(self, cr, e):
        self.bayes = cv2.ml.NormalBayesClassifier_create()
        cr = np.array(cr, dtype=np.float32)
        e = np.array(e, dtype=np.int32)
        self.bayes.train(cr, cv2.ml.ROW_SAMPLE, e)

    def classify(self, xr):
        xr = np.array(xr, dtype=np.float32)
        result = self.bayes.predict(xr)
        return chr(result[1][0][0])
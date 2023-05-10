import cv2
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from classifier.classifier import Classifier


class NaiveBayesClassifier(Classifier):

    def __init__(self):
        self.naive = GaussianNB()

    def classify(self, xr):
        xr = np.array(xr, dtype=np.float32)
        result = self.naive.predict(xr)
        return chr(result[0])

    def create(self, cr, e):
        cr = np.array(cr, dtype=np.float32)
        e = np.array(e, dtype=np.int32)
        self.naive.fit(cr, e)

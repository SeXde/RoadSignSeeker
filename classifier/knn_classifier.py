import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from classifier.classifier import Classifier


class KnnClassifier(Classifier):

    def __init__(self):
        self.knn = KNeighborsClassifier(n_neighbors=3)

    def create(self, cr, e):
        cr = np.array(cr, dtype=np.float32)
        e = np.array(e, dtype=np.int32)
        self.knn.fit(cr, e)

    def classify(self, xr):
        xr = np.array(xr, dtype=np.float32)
        result = self.knn.predict(xr)
        return chr(result[0])

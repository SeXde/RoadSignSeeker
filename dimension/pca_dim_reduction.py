from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from dimension.dim_reduction import DimReduction


class PcaDimReduction(DimReduction):

    def __init__(self):
        self.pca = PCA(n_components=20)

    def create(self, c, e):
        self.pca.fit(c)

    def reduce(self, c) -> []:
        cr = self.pca.transform(c)
        return cr

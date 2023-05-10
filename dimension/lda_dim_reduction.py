from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from dimension.dim_reduction import DimReduction


class LdaDimReduction(DimReduction):

    def __init__(self):
        self.lda = LinearDiscriminantAnalysis()

    def create(self, c, e):
        self.lda.fit(c, e)

    def reduce(self, c) -> []:
        cr = self.lda.transform(c)
        return cr

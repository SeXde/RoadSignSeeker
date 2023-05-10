from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from dimension.dim_reduction import DimReduction


class LdaDimReduction(DimReduction):

    def reduce(self, c, e) -> []:
        lda = LinearDiscriminantAnalysis()
        lda.fit(c, e)
        cr = lda.transform(c)
        return cr

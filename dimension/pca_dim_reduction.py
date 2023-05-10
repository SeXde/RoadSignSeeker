from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from dimension.dim_reduction import DimReduction


class PcaDimReduction(DimReduction):

    def reduce(self, c, lda) -> []:
        cr = lda.transform(c)
        return cr

from surprise import NMF
from com.wday.ml.classifier import Classifier

class NMFMatrixFactorization(Classifier):

    def __init__(self):
        super().__init__("nmf", NMF, param_grid={'n_factors': [15,20], 'n_epochs': [50,70]})
        best_params = super().tune()
        print(best_params)
        self.algo = NMF(n_factors=best_params['n_factors'], n_epochs=best_params['n_epochs'])
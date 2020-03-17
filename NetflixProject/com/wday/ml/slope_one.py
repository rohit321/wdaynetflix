from surprise import SlopeOne
from com.wday.ml.classifier import Classifier

class SlopeOneMatrixFactorization(Classifier):

    def __init__(self):
        super().__init__("slope", SlopeOne, param_grid={})
        best_params = super().tune()
        print(best_params)
        res = not best_params
        if res:
            self.algo = SlopeOne()

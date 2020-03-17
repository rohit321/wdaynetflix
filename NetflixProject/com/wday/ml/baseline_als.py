from surprise import BaselineOnly
from com.wday.ml.classifier import Classifier

class BaseLineALS(Classifier):

    def __init__(self):
        options = {
            'method': 'als',
            'n_epochs': 10,
            'reg_u': 20,
            'reg_i': 15
        }
        super().__init__("bals", BaselineOnly, param_grid={'bsl_options':{'method':['als'], 'n_epochs': [10,15], 'reg_u':[10,20], 'reg_i': [15,25]}})
        best_params = super().tune()
        print(best_params)
        self.algo = BaselineOnly(bsl_options=best_params['bsl_options'])

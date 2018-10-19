# from torchsample.metrics import Metric
from sklearn.metrics import f1_score
from ignite.metrics.metric import Metric
import numpy as np
from hpa_src.data.functional import preds2onehot

# class F1Score(object):
#     def __init__(self, average='macro'):
#         self.average = average
    
#     def __call__(self, y_pred, y_true):
#         return f1_score(y_true, y_pred)
    
#     def reset(self):
#         self.average = 'macro'

class F1Score(Metric):
    """
    Calculates the categorical accuracy.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...)
    - `y` must be in the following shape (batch_size, ...)
    """
    
    def reset(self):
        self.f1 = []
    
    def update(self, output):
        ''' called each batch
        '''
        y_pred, y_true = output
        y_pred = preds2onehot(y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        self.f1.append(f1)
    
    def compute(self):
        if len(self.f1) == 0:
            raise NotComputableError('No f1 computed')
        return np.mean(self.f1)
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
    
import tensorflow as tf
import keras.backend as K

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

import numpy as np
from sklearn.metrics import f1_score

# Samples
y_true = np.array([[1,1,0,0,1], [1,0,1,1,0], [0,1,1,0,0]])
y_pred = np.array([[0,1,1,1,1], [1,0,0,1,1], [1,0,1,0,0]])

# print('Shape y_true:', y_true.shape)
# print('Shape y_pred:', y_pred.shape)

# # Results
# print('sklearn Macro-F1-Score:', f1_score(y_true, y_pred, average='macro'))
# print('Custom Macro-F1-Score:', K.eval(f1(y_true, y_pred)))

# https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras


import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class Metrics(Callback):
    
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print("— val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
        return
 
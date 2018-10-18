import numpy as np

def preds2label(preds, threshold=0, onehot=True):
    ''' Convert prediction to multilabel
    Args:
        preds: prediction, default logits
        threshod: 0 for logits, 0.5 for probs
    '''
    label = np.zeros(preds.shape)

    for i in range(preds.shape[0]):
        lb = np.argwhere(preds[i,:] > threshold)
        if onehot:
            label[i,lb] = 1
            return label
        else:
            yield np.array(lb).flatten()
            
            
def array2str(arr):
    for i in arr:
        yield ' '.join([str(l) for l in i])
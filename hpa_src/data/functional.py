import numpy as np

def preds2onehot(preds, threshold=0):
    ''' Convert prediction to multilabel
    Args:
        preds: prediction, default logits
        threshod: 0 for logits, 0.5 for probs
    '''
    label = preds > threshold
    return label
            
def preds2label(preds, threshold=0, fill_na=False):
    ''' Convert prediction to multilabel
    Args:
        preds: prediction, default logits
        threshod: 0 for logits, 0.5 for probs
    '''
    label = np.zeros(preds.shape)

    for i in range(preds.shape[0]):
        lb = np.argwhere(preds[i,:] > threshold)
        lb = list(lb.flatten())
        if len(lb) == 0 and fill_na: # give the most likely one
            lb = [np.argmax(preds[i,:])]
        yield np.array(lb).flatten()
            
def array2str(arr):
    for i in arr:
        yield ' '.join([str(l) for l in i])

        
def optim_threshold(y_true, y_pred):
    ''' Optimize threshold for a single class
    '''
    scores = []
    thrs = np.arange(0.01,0.9,0.01)
    for p in thrs:
        scores.append(f1_score(y_true, y_pred>p))
    return thrs[np.array(scores).argmax()]


def apply_threshold(prediction, threshold):
    pred = np.stack([prediction[:,i] > threshold[i] for i in range(prediction.shape[1])])
    pred = pred.T
    for i in range(pred.shape[0]):
        lb = np.argwhere(pred[i,:])
        lb = list(lb.flatten())
        yield np.array(lb).flatten()

def iterable_cycle(iterable, input_torch=True):
    """
    Args:
      iterable: object with an __iter__ method that can be called multiple times
      input_torch: input torch tensor
    """
    while True:
        for item in iterable:
            if input_torch:
                x, y = item
                yield x.numpy(), y.numpy()
            else:
                yield x
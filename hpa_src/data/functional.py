import numpy as np

def preds2onehot(preds, threshold=0):
    ''' Convert prediction to multilabel
    Args:
        preds: prediction, default logits
        threshod: 0 for logits, 0.5 for probs
    '''
    label = np.zeros(preds.shape)

    for i in range(preds.shape[0]):
        lb = np.argwhere(preds[i,:] > threshold)
        label[i,lb] = 1
    return label
            
def preds2label(preds, threshold=0):
    ''' Convert prediction to multilabel
    Args:
        preds: prediction, default logits
        threshod: 0 for logits, 0.5 for probs
    '''
    label = np.zeros(preds.shape)

    for i in range(preds.shape[0]):
        lb = np.argwhere(preds[i,:] > threshold)
        yield np.array(lb).flatten()
            
def array2str(arr):
    for i in arr:
        yield ' '.join([str(l) for l in i])


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
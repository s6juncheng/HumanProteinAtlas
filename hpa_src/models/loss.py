from torch.nn.modules.loss import _WeightedLoss
import torch.nn as nn
import torch

class CrossEntropyLossOneHot(_WeightedLoss):
    '''
    nn.LogSoftmax(x): compute element log(q), q is the predicted probabiliy, input x is the predicted logit
    '''
    def __init__(self, weight=None, size_average=True, ignore_index=-100,
             reduce=None, reduction='elementwise_mean'):
        super(CrossEntropyLossOneHot, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
    
    def forward(self, input, target):
        logsoftmax = nn.LogSoftmax(1)
        if self.weight is None:
            return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
        else:
            return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
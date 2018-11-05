from torch.nn.modules.loss import _WeightedLoss
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

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
        p_hat = logsoftmax(input) # log(p_hat)
        p_hat_1 = logsoftmax(-input)
        if self.weight is None:
            return torch.mean(-target*p_hat - (1-target)*(p_hat_1), dim=1)
        else:
            return torch.mean(-target*p_hat-(1-target)*(p_hat_1), dim=1)
        
        
class FocalLoss(nn.Module):
    def __init__(self, 
                 alpha=None,
                 gamma=2,
                 logits=False, 
                 reduction='elementwise_mean'):
        
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        if self.alpha is not None:
            self.alpha = torch.from_numpy(alpha).float()
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
            
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none') # -log(p_t), p_t: predicted target prob
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = torch.pow((1-pt), self.gamma) * BCE_loss
        if self.alpha is not None:
            if inputs.is_cuda and not self.alpha.is_cuda:
                device = inputs.device
                self.alpha = self.alpha.cuda(device)
            F_loss = self.alpha * F_loss

        if self.reduction == 'none':
            return F_loss
        elif self.reduction == 'elementwise_mean':
            return torch.mean(F_loss)
        else:
            return F_loss.sum()
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Sachin Mehta"


class CrossEntropyLoss2d(nn.Module):
    '''
    This file defines a cross entropy loss for 2D images
    '''
    def __init__(self, weight=None):
        '''
        :param weight: 1D weight vector to deal with the class-imbalance
        '''
        super().__init__()

        self.loss = nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, 1), targets)

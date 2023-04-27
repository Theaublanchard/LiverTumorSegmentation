import numpy as np
import torch

class MetricLogger(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
    


def dice_loss(pred,target,gamma=1):
    '''
    Implementation of dice loss for multi class segmentation

    Args:
        pred: torch.Tensor 
            Predicted segmentation mask of shape (batch_size, n_classes, height, width)
        target: torch.Tensor
            Ground truth segmentation mask of shape (batch_size, n_classes, height, width)
        gamma: float
            Smoothing factor to avoid division by zero

    Returns:    
        dice_loss: torch.Tensor
            Dice loss for the given batch
    '''
    num = (target*pred).sum(dim=(0,2,3)) + gamma
    denum = (target*target).sum(dim=(0,2,3)) + (pred*pred).sum(dim=(0,2,3)) + gamma
    return (1 - 2*num/denum).mean()


def cos_sim(a,b,eps = 1e-8):
    '''
    Computes the cosine similarity between two images channel-wise
    TODO : change value to one when a channel is all zeros
    Args:
        a: torch.Tensor
            Vector of shape (B,C, H, W)
        b: torch.Tensor
            Vector of shape (B,C, H, W)
    '''
    a_reshaped = a.reshape(3,-1)
    b_reshaped = b.reshape(3,-1)
    num = (a_reshaped @ b_reshaped.T).diag()
    denum = a_reshaped.norm(dim=1)*b_reshaped.norm(dim=1)
    return (num/(denum + eps))

def t_vMF_dice_loss(pred,target,k):
    '''  
    Implementation of the t-vMF dice loss for multi class segmentation

    Args:
        pred: torch.Tensor
            Predicted segmentation mask of shape (batch_size, n_classes, height, width)
        target: torch.Tensor
            Ground truth segmentation mask of shape (batch_size, n_classes, height, width)
        k: torch.Tensor
            Concentration parameter of the t-vMF distribution of shape (n_classes,)
    
    Returns:
        t_vMF_dice_loss: torch.Tensor
    '''
    cos = cos_sim(pred,target)
    num = 1 + cos
    denum = 1 + k*(1-cos)
    phi = num/denum - 1
    return torch.square(1-phi).mean()



def dice_score(pred,target,eps=1e-8):
    '''
    Computes the dice score channel wise

    Args:
        pred: torch.Tensor
            Predicted segmentation mask of shape (batch_size, n_classes, height, width)
        target: torch.Tensor
            Ground truth segmentation mask of shape (batch_size, n_classes, height, width)

    Returns:
        dice_score: torch.Tensor
            Dice score for each channel
    '''

    num = (pred * target).sum(dim=(0,2,3))
    denum = pred.int().sum(dim=(0,2,3)) + target.int().sum(dim=(0,2,3))
    return 2*num/(denum+eps)
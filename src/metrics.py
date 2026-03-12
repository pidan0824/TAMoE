import torch.nn.functional as F

def mse(y_true, y_pred):
    return F.mse_loss(y_true, y_pred, reduction='mean')

def mae(y_true, y_pred):
    return F.l1_loss(y_true, y_pred, reduction='mean')

__all__ = ['PatchCB', 'create_patch', 'RevInCB']

import torch
from .core import Callback
from ..masking.base import create_patch
from ..models.layers.revin import RevIN


class PatchCB(Callback):

    def __init__(self, patch_len, stride):
        """
        Callback used to perform patching on the batch input data.
        Args:
            patch_len:        patch length
            stride:           stride
        """
        self.patch_len = patch_len
        self.stride = stride

    def before_forward(self): self.set_patch()

    def set_patch(self):
        """[bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]"""
        xb_patch, _ = create_patch(self.xb, self.patch_len, self.stride)
        self.learner.xb = xb_patch

class RevInCB(Callback):
    def __init__(self, num_features: int, eps=1e-5, 
                        affine:bool=False, denorm:bool=True):
        """        
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        :param denorm: if True, the output will be de-normalized

        This callback only works with affine=False.
        if affine=True, the learnable affine_weights and affine_bias are not learnt
        """
        super().__init__()
        self.denorm = denorm
        self.revin = RevIN(num_features, eps, affine)
    

    def before_forward(self): self.revin_norm()
    def after_forward(self): 
        if self.denorm: self.revin_denorm() 
        
    def revin_norm(self):
        xb_revin = self.revin(self.xb, 'norm')      # xb_revin: [bs x seq_len x nvars]
        self.learner.xb = xb_revin

    def revin_denorm(self):
        """
        Consistent with original PatchTST: only denormalize pred, not yb
        """
        pred = self.pred
        if isinstance(pred, dict):
            for key in pred:
                if isinstance(pred[key], torch.Tensor):
                    pred[key] = self.revin(pred[key], 'denorm')
        elif isinstance(pred, (tuple, list)) and len(pred) > 0:
            main_pred = pred[0]
            if isinstance(main_pred, torch.Tensor):
                main_pred = self.revin(main_pred, 'denorm')
            if isinstance(pred, tuple):
                pred = (main_pred,) + tuple(pred[1:])
            else:
                pred = [main_pred] + list(pred[1:])
        else:
            pred = self.revin(pred, 'denorm')
        self.learner.pred = pred

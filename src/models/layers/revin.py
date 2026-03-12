import torch
from torch import nn
import warnings

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if not hasattr(self, 'stdev') or self.stdev is None or not hasattr(self, 'mean') or self.mean is None:
            warnings.warn(
                "[RevIN._denormalize] Warning: stdev/mean not initialized, will compute from current input. "
                "This may lead to incorrect denormalization results since input x is already normalized. "
                "Please ensure normalize is called before denormalize.",
                RuntimeWarning
            )
            self._get_statistics(x)
        
        if self.stdev.device != x.device:
            self.stdev = self.stdev.to(x.device)
            self.mean = self.mean.to(x.device)
        if self.stdev.shape != x.shape:
            if self.stdev.ndim < x.ndim:
                while self.stdev.ndim < x.ndim:
                    self.stdev = self.stdev.unsqueeze(-1)
                while self.mean.ndim < x.ndim:
                    self.mean = self.mean.unsqueeze(-1)
        
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

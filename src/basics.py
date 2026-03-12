import torch


class GetAttr:

    "Inherit from this to have all attr accesses in `self._xtra` passed down to `self.default`"
    _default='default'
    def _component_attr_filter(self,k):
        if k.startswith('__') or k in ('_xtra',self._default): return False
        xtra = getattr(self,'_xtra',None)
        return xtra is None or k in xtra

    def _dir(self): 
        return [k for k in dir(getattr(self,self._default)) if self._component_attr_filter(k)]

    def __getattr__(self, k):
        if self._component_attr_filter(k):
            attr = getattr(self, self._default, None)
            if attr is not None: return getattr(attr,k)

    def __dir__(self):
        return list(set(dir(type(self))) | set(self.__dict__.keys()) | set(self._dir()))

    def __setstate__(self,data): 
        self.__dict__.update(data)



def default_device(use_cuda=True):
    "Return or set default device; `use_cuda`: True/None - CUDA if available, else CPU; False - CPU"
    if not torch.cuda.is_available():
        use_cuda = False
    return torch.device(torch.cuda.current_device()) if use_cuda else torch.device('cpu')


def to_device(b, device=None, non_blocking=False):
    "Recursively put `b` on `device`"
    if device is None: 
        device = default_device(use_cuda=True)

    if isinstance(b, dict):
        return {key: to_device(val, device) for key, val in b.items()}

    if isinstance(b, (list, tuple)):        
        if len(b) == 2 and isinstance(b[1], int):
            return (to_device(b[0], device), b[1])
        return type(b)(to_device(o, device) if hasattr(o, 'to') else o for o in b)    
    
    return b.to(device, non_blocking=non_blocking)


def to_numpy(b):
    "Recursively convert tensors to numpy arrays"
    if isinstance(b, dict):
        return {key: to_numpy(val) for key, val in b.items()}

    if isinstance(b, (list, tuple)):
        return type(b)(to_numpy(o) for o in b)

    return b.detach().cpu().numpy()


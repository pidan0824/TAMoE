__all__ = ['Callback', 'SetupLearnerCB', 'GetPredictionsCB', 'GetTestCB' ]


""" 
Callback lists:
    > before_fit
        - before_epoch
            + before_epoch_train                
                ~ before_batch_train
                ~ after_batch_train                
            + after_epoch_train

            + before_epoch_valid                
                ~ before_batch_valid
                ~ after_batch_valid                
            + after_epoch_valid
        - after_epoch
    > after_fit

    - before_predict        
        ~ before_batch_predict
        ~ after_batch_predict          
    - after_predict

"""

from ..basics import *
import torch

class Callback(GetAttr): 
    _default='learner'

    def _get_model_and_backbone(self):
        """Get (actual_model, backbone), unwrapping probe wrapper.

        Returns:
            (actual_model, backbone) where backbone may be None.
        """
        model = getattr(self.learner, "model", None)
        if model is None:
            return None, None
        if hasattr(model, "model") and hasattr(model.model, "backbone"):
            model = model.model
        backbone = getattr(model, "backbone", None)
        if backbone is None and hasattr(model, 'get_moe_aux_loss'):
            backbone = model
        return model, backbone

    def _get_backbone(self):
        """Get backbone model, unwrapping probe wrapper."""
        _, backbone = self._get_model_and_backbone()
        return backbone


class SetupLearnerCB(Callback): 
    def __init__(self):        
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{torch.cuda.current_device()}')
        else:
            self.device = torch.device('cpu')

    def before_batch_train(self): self._to_device()
    def before_batch_valid(self): self._to_device()
    def before_batch_predict(self): self._to_device()
    def before_batch_test(self): self._to_device()

    def _to_device(self):
        batch = to_device(self.batch, self.device)
        if self.n_inp > 1: xb, yb = batch
        else: xb, yb = batch, None
        self.learner.batch = xb, yb
        
    def before_fit(self):
        if next(self.learner.model.parameters()).device != self.device:
            self.learner.model.to(self.device)
        self.learner.device = self.device                        


class GetPredictionsCB(Callback):
    def before_predict(self):
        self.preds = []        
    
    def after_batch_predict(self):        
        self.preds.append(self.pred)

    def after_predict(self):
        self.preds = torch.concat(self.preds)

         

class GetTestCB(Callback):
    def before_test(self):
        self.preds, self.targets = [], []        
    
    def after_batch_test(self):
        self.preds.append(self.pred)
        self.targets.append(self.yb)

    def after_test(self):
        self.preds = torch.concat(self.preds)
        self.targets = torch.concat(self.targets)

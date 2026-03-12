from typing import List
import torch
from torch.optim import Adam
from torch import nn
from .basics import *
from .callback.core import * 
from .callback.tracking import * 
from .callback.scheduler import *
from .utils import *
import os
import numpy as np


class Learner(GetAttr):

    # =====================================================================
    # Initialization & Setup
    # =====================================================================

    def __init__(self, dls, model, 
                        loss_func=None, 
                        lr=1e-3, 
                        cbs=None, 
                        metrics=None, 
                        opt_func=Adam,
                        **kwargs):
                
        self.model, self.dls, self.loss_func, self.lr = model, dls, loss_func, lr
        self.opt_func = opt_func
        self.set_opt()
        
        self.metrics = metrics
        self.n_inp  = 2
        self.forward_kwargs = {}
        if cbs and not isinstance(cbs, List): cbs = [cbs]
        self.initialize_callbacks(cbs)
        self.run_finder = False

    def set_opt(self):
        if self.model:
            self.opt = self.opt_func(self.model.parameters(), self.lr)
        else: self.opt = None

    # =====================================================================
    # Properties
    # =====================================================================

    @property
    def callbacks(self):
        "Backward-compatible alias for self.cbs"
        return getattr(self, 'cbs', None)

    @callbacks.setter
    def callbacks(self, value):
        self.cbs = value

    # =====================================================================
    # Callback Management
    # =====================================================================

    def default_callback(self):
        "get a set of default callbacks"
        default_cbs = [ SetupLearnerCB(), TrackTimerCB(), 
                        TrackTrainingCB(train_metrics=False, valid_metrics=True)]                  
        return default_cbs

    def initialize_callbacks(self, cbs):
        default_cbs = self.default_callback()
        self.cbs = update_callbacks(cbs, default_cbs) if cbs else default_cbs
        self.cbs += [PrintResultsCB()]
        for cb in self.cbs: cb.learner = self
        self('init_cb')       

    def add_callback(self, cb):                
        if not cb: return
        cb.learner = self
        self.cbs = update_callback(cb, self.cbs)           

    def add_callbacks(self, cbs):        
        if not isinstance(cbs, list):  cbs = [cbs]
        for cb in cbs: self.add_callback(cb)

    def remove_callback(self, cb): 
        cb.learner = None
        self.cbs, removed_cb = remove_callback(cb, self.cbs)
        return removed_cb
    
    def remove_callbacks(self, cb_list):
        for cb in cb_list: self.remove_callback(cb)

    def __call__(self, name):        
        for cb in self.cbs: 
            attr = getattr(cb, name)
            if attr is not None: attr()

    # =====================================================================
    # Training Interface (high-level)
    # =====================================================================

    def fit(self, n_epochs, cbs=None, do_valid=True):
        " fit the model "
        self.n_epochs = n_epochs
        if not self.dls.valid: do_valid = False
        if cbs: self.add_callbacks(cbs)

        # Call before_fit first to initialize components (e.g., task token generator)
        self('before_fit')

        try:
            for self.epoch in range(n_epochs):
                self('before_epoch')
                self.one_epoch(train=True)
                if do_valid: self.one_epoch(train=False)
                self('after_epoch')
        except KeyboardInterrupt: pass 
        self('after_fit')

    def fit_one_cycle(self, n_epochs, lr_max=None, pct_start=0.3):
        self.n_epochs = n_epochs        
        self.lr_max = lr_max if lr_max else self.lr
        cb = OneCycleLR(lr_max=self.lr_max, pct_start=pct_start)
        self.fit(self.n_epochs, cbs=cb)                

    def fine_tune(self, n_epochs, base_lr=None, freeze_epochs=1, pct_start=0.3):
        """
        Finetune the pretrained model. First the entire model is frozen, only head is trained
        up to a freeze_epochs number. Then the model is unfrozen and the entire model is trained.
        """
        assert (n_epochs>0)|(freeze_epochs>0), "Either n_epochs or freeze_epochs has to be > 0"
        if base_lr is None: base_lr = self.lr
        # Finetune the head if freeze_epochs > 0:
        if freeze_epochs > 0:
            print('Finetune the head')
            self.freeze()
            self.fit_one_cycle(freeze_epochs, lr_max=base_lr, pct_start=pct_start)
        
        # Finetune the entire network if n_epochs > 0
        if n_epochs > 0:
            print('Finetune the entire network')        
            self.unfreeze()
            # Reset TrackerCB/SaveModelCB.best so phase 2 tracks its own best,
            # instead of requiring it to beat phase 1's best (which only trained head).
            for cb in self.cbs:
                if hasattr(cb, 'best') and hasattr(cb, 'monitor'):
                    cb.best = None
            self.fit_one_cycle(n_epochs, lr_max=base_lr/2, pct_start=pct_start)

    def linear_probe(self, n_epochs, base_lr=None, pct_start=0.3):
        """
        Linear probing the pretrained model. The model is frozen except the head during finetuning.
        """
        assert (n_epochs>0), "n_epochs has to be > 0"
        if base_lr is None: base_lr = self.lr
        print('Finetune the head')
        self.freeze()
        self.fit_one_cycle(n_epochs, lr_max=base_lr, pct_start=pct_start)

    def lr_finder(self, start_lr=1e-7, end_lr=10, num_iter=100, step_mode='exp',
                   suggestion='valley'):
        """
        Find optimal learning rate using valley method.
        """
        n_epochs = num_iter//len(self.dls.train) + 1
        self.run_finder = True
        cb = LRFinderCB(start_lr, end_lr, num_iter, step_mode, suggestion=suggestion)
        self.fit(n_epochs=n_epochs, cbs=cb, do_valid=False)
        self.remove_callback(cb)
        self.run_finder = False
        
        if suggestion:
            return cb.suggested_lr

    # =====================================================================
    # Training Loop (internal)
    # =====================================================================

    def one_epoch(self, train):                           
        self.epoch_train() if train else self.epoch_validate()        

    def epoch_train(self):
        self('before_epoch_train')
        self.model.train()                
        self.dl = self.dls.train
        self.all_batches('train')
        self('after_epoch_train')
    
    def epoch_validate(self, dl=None):
        self('before_epoch_valid')
        self.model.eval()
        self.dl = dl if dl else self.dls.valid
        if self.dl:        
            with torch.no_grad(): self.all_batches('valid')
        self('after_epoch_valid')

    _BATCH_DISPATCH = {
        'train':   ('before_batch_train',   '_do_batch_train',    'after_batch_train'),
        'valid':   ('before_batch_valid',   '_do_batch_validate', 'after_batch_valid'),
        'predict': ('before_batch_predict', '_do_batch_predict',  'after_batch_predict'),
        'test':    ('before_batch_test',    '_do_batch_test',     'after_batch_test'),
    }

    def all_batches(self, type_):
        before, do_fn, after = self._BATCH_DISPATCH[type_]
        for num, batch in enumerate(self.dl):
            self.iter, self.batch = num, batch
            self(before)
            getattr(self, do_fn)()
            self(after)

    # =====================================================================
    # Batch Processing
    # =====================================================================

    def model_forward(self):
        self('before_forward')
        fwk = getattr(self, "forward_kwargs", {}) or {}
        self.pred = self.model(self.xb, **fwk)
        self('after_forward')
        return self.pred

    def _forward_and_loss(self, batch):
        """Forward pass + loss with callback hooks (shared by train/valid)."""
        self.xb, self.yb = batch
        pred = self.model_forward()
        self.loss = None
        self('before_loss')
        if self.loss is None:
            loss = self.loss_func(pred, self.yb)
        else:
            loss = self.loss
        self('after_loss')
        return pred, loss

    def _forward_step(self, batch):
        """Forward pass only (no loss). Sets self.xb, self.yb."""
        self.xb, self.yb = batch
        return self.model_forward()

    def _do_batch_train(self):
        self.pred, self.loss = self.train_step(self.batch)
        self.opt.zero_grad()
        self('before_backward')
        self.loss.backward()
        self.opt.step()

    def train_step(self, batch):
        return self._forward_and_loss(batch)

    def _do_batch_validate(self):
        self.pred, self.loss = self.valid_step(self.batch)

    def valid_step(self, batch):
        return self._forward_and_loss(batch)

    def _do_batch_predict(self):
        self.pred = self.predict_step(self.batch)

    def predict_step(self, batch):
        return self._forward_step(batch)

    def _do_batch_test(self):
        self.pred = self.test_step(self.batch)

    def test_step(self, batch):
        return self._forward_step(batch)

    # =====================================================================
    # Prediction & Testing 
    # =====================================================================

    def predict(self, test_data, weight_path=None, Dataset=None, Dataloader=None, batch_size=None):
        """
        Run inference on test_data (tensor, numpy array, dataset, or dataloader).
        Returns predictions as a numpy array.
        """                
        if weight_path is not None: self.load(weight_path)
        cb = GetPredictionsCB()
        self.add_callback(cb)                    
        test_dl = self._prepare_data(test_data, Dataset, Dataloader, batch_size)
        self._predict(test_dl)        
        self.preds = cb.preds
        return to_numpy(self.preds) 

    def _predict(self, dl=None):
        self('before_predict')
        if dl is None: return
        self.dl = dl
        self.n_inp = dl.dataset.n_inp
        self.model.eval()
        with torch.no_grad(): self.all_batches('predict')        
        self('after_predict')

    def test(self, dl, weight_path=None, scores=None):
        """
        Run test on a dataloader and optionally compute scores.
        Returns predictions and targets (and scores if provided).
        """          
        if dl is None: return
        else: self.dl = dl
        if weight_path is not None: self.load(weight_path)
        cb = GetTestCB()
        self.add_callback(cb)
        self('before_test')
        self.model.eval()
        with torch.no_grad(): self.all_batches('test')
        self('after_test')   
        self.preds, self.targets = to_numpy([cb.preds, cb.targets])
        if scores: 
            s_vals = [score(cb.targets, cb.preds).to('cpu').numpy() for score in list(scores)]
            return self.preds, self.targets, s_vals
        else: return self.preds, self.targets

    def _prepare_data(self, test_data, Dataset=None, Dataloader=None, batch_size=None):
        if test_data is None: return test_data
        if Dataset and Dataloader:
            test_dset = Dataset(test_data)
            if not batch_size: batch_size=16
            test_dl = Dataloader(test_dset, batch_size)        
        else:            
            if self.dls: 
                # add test_data to the dataloader defined in the dls.train
                test_dl = self.dls.add_dl(test_data, batch_size=batch_size)  
            else: test_dl = test_data       # assume test_data is already a form of dataloader
        return test_dl

    # =====================================================================
    # Model State (freeze / unfreeze)
    # =====================================================================

    def freeze(self):
        """
        Freeze backbone, only train head.
        Requires the model to have 'head' attribute.
        """
        if not hasattr(self.model, 'head'):
            raise AttributeError("[Learner.freeze] Model has no 'head' attribute")
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.model.parameters(): param.requires_grad = True        

    # =====================================================================
    # Persistence (save / load)
    # =====================================================================

    def save(self, fname, path, **kwargs):
        """
        Save model and optimizer state to `path/fname.pth`.
        """
        fname = join_path_file(fname, path, ext='.pth')
        model_config = kwargs.pop('model_config', None)
        save_model(fname, self.model, getattr(self,'opt',None), model_config=model_config, **kwargs)
        return fname

    def load(self, fname, with_opt=False, device='cuda', strict=True, **kwargs):
        """
        load the model
        """
        if not torch.cuda.is_available():
            device = "cpu"
        load_model(fname, self.model, self.opt, with_opt, device=device, strict=strict)


def update_callback(cb, list_cbs):
    list_cbs = [cb_ for cb_ in list_cbs if type(cb_) != type(cb)]
    list_cbs.append(cb)
    return list_cbs

def update_callbacks(list_cbs, default_cbs):
    for cb in list_cbs: default_cbs = update_callback(cb, default_cbs)
    return default_cbs

def remove_callback(cb, list_cbs):
    removed = None
    for cb_ in list_cbs:
        if type(cb_) == type(cb):
            list_cbs.remove(cb_)
            removed = cb_
            break
    return list_cbs, removed


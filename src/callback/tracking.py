__all__ = ['TrackTimerCB', 'TrackTrainingCB', 'PrintResultsCB', 'TerminateOnNaNCB',
            'TrackerCB', 'SaveModelCB']

from ..basics import *
from .core import Callback
import torch
import time
import numpy as np


class TrackTimerCB(Callback):
    def before_fit(self):
        self.learner.epoch_time = None

    def before_epoch_train(self):         
        self.start_time = time.time()

    def after_epoch_train(self): 
        self.learner.epoch_time = self.format_time(time.time() - self.start_time)

    def format_time(self, t):
        "Format `t` (in seconds) to (h):mm:ss"
        t = int(t)
        h, m, s = t // 3600, (t // 60) % 60, t % 60
        if h != 0:
            return f'{h}:{m:02d}:{s:02d}'
        else:
            return f'{m:02d}:{s:02d}'


class TrackTrainingCB(Callback):

    def __init__(self, train_metrics=False, valid_metrics=True):
        super().__init__()        
        self.train_metrics, self.valid_metrics = train_metrics, valid_metrics 

    def before_fit(self):
        self.setup()
        self.initialize_recorder()
        self.mean_reduction_ = True
        if hasattr(self.loss_func, 'reduction'):
            self.mean_reduction_ = (self.loss_func.reduction == 'mean')
    
    def setup(self):
        self.valid_loss = False
        if self.learner.dls: 
            if not self.learner.dls.valid: self.valid_metrics = False    
            else: self.valid_loss = True

        if self.metrics:
            if not isinstance(self.metrics, list): self.metrics = [self.metrics]   
            self.metric_names = [func.__name__ for func in self.metrics]                       
        else: self.metrics, self.metric_names = [], []        
            
    def initialize_recorder(self):
        recorder = {'epoch': [],  'train_loss': []} 
        if self.valid_loss: recorder['valid_loss'] = []

        for name in self.metric_names: 
            if self.train_metrics: recorder['train_'+name] = []            
            if self.valid_metrics: recorder['valid_'+name] = []
        self.recorder = recorder        
        self.learner.recorder = recorder

    def initialize_batch_recorder(self, with_metrics):
        batch_recorder = {'n_samples': [], 'batch_losses': [], 'with_metrics': with_metrics}                                                         
        self.batch_recorder = batch_recorder

    def reset(self):
        self.targs, self.preds = [],[]
        self.n_samples = 0


    def after_epoch(self):
        self.recorder['epoch'].append(self.epoch)              
        
    def before_epoch_train(self):
        self.initialize_batch_recorder(with_metrics=self.train_metrics)
        self.reset()

    def before_epoch_valid(self):
        self.initialize_batch_recorder(with_metrics=self.valid_metrics)
        self.reset()


    def after_epoch_train(self):
        values = self.compute_scores()
        self.recorder['train_loss'].append( values['loss'] )
        if self.train_metrics:
            for name, func in zip(self.metric_names, self.metrics): 
                self.recorder['train_'+name].append( values[name] )

    def after_epoch_valid(self):
        if not self.learner.dls.valid: return
        values = self.compute_scores()
        self.recorder['valid_loss'].append( values['loss'] )
        if self.valid_metrics:
            for name, func in zip(self.metric_names, self.metrics): 
                self.recorder['valid_'+name].append( values[name] )

    def after_batch_train(self): self.accumulate()
    def after_batch_valid(self): self.accumulate()
        
    def accumulate(self ):
        xb, batch_yb = self.batch
        bs = len(xb)
        self.batch_recorder['n_samples'].append(bs)
        loss = self.loss.detach()*bs if self.mean_reduction_ else self.loss.detach()
        self.batch_recorder['batch_losses'].append(loss)

        yb = getattr(self.learner, 'yb', batch_yb)

        if yb is None: self.batch_recorder['with_metrics'] = False
        if len(self.metrics) == 0: self.batch_recorder['with_metrics'] = False
        if self.batch_recorder['with_metrics']:
            self.preds.append(self.pred.detach().cpu())
            self.targs.append(yb.detach().cpu())
    

    def compute_scores(self):
        "calculate losses and metrics after each epoch"
        values = {}
        n = sum(self.batch_recorder['n_samples'])
        values['loss'] = sum(self.batch_recorder['batch_losses']).item()/n

        if len(self.preds) == 0: return values
        self.preds = torch.cat(self.preds)
        self.targs = torch.cat(self.targs)
        for func in self.metrics:
            values[func.__name__] = func(self.targs, self.preds)
        return values
    

class TerminateOnNaNCB(Callback):
    " A callback to stop the training if loss is NaN"
    def after_batch_train(self):
        if torch.isinf(self.loss) or torch.isnan(self.loss): raise KeyboardInterrupt


class PrintResultsCB(Callback):
    def _sort_header(self, header):
        ordered_header = []
        loss_keys = []
        metric_keys = []
        other_keys = []
        
        for key in header:
            if key == 'epoch':
                ordered_header.insert(0, key)
            elif 'loss' in key.lower():
                loss_keys.append(key)
            elif key in ['time']:
                other_keys.append(key)
            else:
                metric_keys.append(key)
        
        loss_keys_sorted = []
        if 'train_loss' in loss_keys:
            loss_keys_sorted.append('train_loss')
            loss_keys.remove('train_loss')
        if 'valid_loss' in loss_keys:
            loss_keys_sorted.append('valid_loss')
            loss_keys.remove('valid_loss')
        loss_keys_sorted.extend(sorted(loss_keys))
        
        ordered_header.extend(loss_keys_sorted)
        ordered_header.extend(sorted(metric_keys))
        ordered_header.extend(other_keys)
        return ordered_header

    def before_fit(self):
        if self.run_finder: return
        if not hasattr(self.learner, 'recorder'): return
        header = list(self.learner.recorder.keys()) + ['time']
        header = self._sort_header(header)
        print('{:>15s}'.format(header[0]) + ''.join('{:>15s}'.format(h) for h in header[1:]))
    
    def after_epoch(self):      
        if self.run_finder: return
        if not hasattr(self.learner, 'recorder'): return
        
        header = self._sort_header(list(self.learner.recorder.keys()))
        
        epoch_logs = []
        for key in header:
            value = self.learner.recorder[key][-1] if self.learner.recorder[key] else None            
            epoch_logs.append(value)
        if self.learner.epoch_time and 'time' not in header: 
            epoch_logs.append(self.learner.epoch_time)
            header.append('time')
        
        formatted_values = []
        for i, val in enumerate(epoch_logs):
            if i == 0:  
                formatted_values.append(f'{val:>15d}' if val is not None else f'{"":>15s}')
            elif i == len(epoch_logs) - 1 and self.learner.epoch_time and header[i] == 'time':  # time (string)
                formatted_values.append(f'{val:>15s}' if val is not None else f'{"":>15s}')
            else:  
                formatted_values.append(f'{val:>15.6f}' if val is not None else f'{"":>15s}')
        print(''.join(formatted_values))


class TrackerCB(Callback):
    def __init__(self, monitor='train_loss', comp=None, min_delta=0.):
        super().__init__()
        if comp is None: comp = np.less if 'loss' in monitor or 'error' in monitor else np.greater
        if comp == np.less: min_delta *= -1
        self.monitor, self.comp, self.min_delta = monitor, comp, min_delta
        self.best = None

    def before_fit(self):
        if self.run_finder: return
        if self.best is None: self.best = float('inf') if self.comp == np.less else -float('inf')
        self.monitor_names = list(self.learner.recorder.keys())
        assert self.monitor in self.monitor_names

    def after_epoch(self):        
        if self.run_finder: return
        val = self.learner.recorder[self.monitor][-1]
        if self.comp(val - self.min_delta, self.best): self.best, self.new_best = val,True
        else: self.new_best = False


class SaveModelCB(TrackerCB):
    def __init__(self, monitor='train_loss', comp=None, min_delta=0., 
                        every_epoch=False, fname='model', path=None, with_opt=False, save_process_id=0, global_rank=None,
                        extra_save_fn=None, load_best_after_fit=True):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta)        
        self.every_epoch = every_epoch
        self.last_saved_path = None
        self.path, self.fname = path, fname
        self.with_opt = with_opt
        self.save_process_id = save_process_id
        self.extra_save_fn = extra_save_fn
        self.load_best_after_fit = load_best_after_fit

       
        if global_rank:
            self.global_rank = int(global_rank)
        else:
            if torch.cuda.is_available():
                self.global_rank = torch.cuda.current_device()
                if not torch.distributed.is_initialized():
                    self.save_process_id = self.global_rank
            else:
                self.global_rank = 0


    def _save(self, fname, path):
        if self.global_rank == self.save_process_id:
            model = self.learner.model
            model_config = getattr(model, 'config', None)
            saved_path = self.learner.save(fname, path, with_opt=self.with_opt, model_config=model_config)
            
            if saved_path:
                print(f"[SaveModelCB] Checkpoint saved to: {saved_path}")
            if saved_path and self.extra_save_fn is not None:
                try:
                    self.extra_save_fn(fname, path)
                except Exception as exc:
                    print(f'[SaveModelCB] Warning: extra_save_fn failed with error: {exc}')
            return saved_path

    def after_epoch(self):
        if self.every_epoch:
            if ((self.epoch%self.every_epoch) == 0) or (self.epoch==self.n_epochs-1): 
                self._save(f'{self.fname}_{self.epoch}', self.path)
        else:
            super().after_epoch()
            if self.new_best:
                print(f'Better model found at epoch {self.epoch} with {self.monitor} value: {self.best}.')
                saved_path = self._save(f'{self.fname}', self.path)
                self.last_saved_path = saved_path

    def after_fit(self):
        if self.run_finder: return
        if (not self.every_epoch and self.global_rank == self.save_process_id 
                and self.load_best_after_fit and self.last_saved_path):
            self.learner.load(self.last_saved_path, with_opt=self.with_opt)

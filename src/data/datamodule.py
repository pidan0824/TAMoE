import warnings
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset


def create_worker_init_fn(base_seed):
    """
    Create a worker_init_fn that uses explicit base_seed.
    
    This is more reliable than using torch.initial_seed() which may not be
    set correctly if seed is set after DataLoader creation.
    
    Args:
        base_seed: Base seed value (will be modified per worker)
    
    Returns:
        worker_init_fn function
    """
    def worker_init_fn(worker_id):
        """Set random seed for each DataLoader worker to ensure reproducibility."""
        # Use base_seed + worker_id to ensure each worker has different but deterministic seed
        worker_seed = (base_seed + worker_id) % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(worker_seed)
            torch.cuda.manual_seed_all(worker_seed)
    return worker_init_fn


class DataLoaders:
    def __init__(
        self,
        datasetCls,
        dataset_kwargs: dict,
        batch_size: int,
        workers: int = 0,
        collate_fn=None,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        seed: int = None,
    ):
        super().__init__()
        self.datasetCls = datasetCls
        self.batch_size = batch_size

        if "split" in dataset_kwargs.keys():
            del dataset_kwargs["split"]
        self.dataset_kwargs = dataset_kwargs
        self.workers = workers
        self.collate_fn = collate_fn
        self.shuffle_train, self.shuffle_val = shuffle_train, shuffle_val

        # Store seed for worker initialization
        self.seed = seed
        
        # Create generator for DataLoader shuffle to ensure reproducibility
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        
        # Create worker_init_fn with explicit seed
        self.worker_init_fn = create_worker_init_fn(seed) if seed is not None else None

        self.train = self.train_dataloader()
        self.valid = self.val_dataloader()
        self.test = self.test_dataloader()

    def train_dataloader(self):
        return self._make_dloader("train", shuffle=self.shuffle_train)

    def val_dataloader(self):
        return self._make_dloader("val", shuffle=self.shuffle_val)

    def test_dataloader(self):
        return self._make_dloader("test", shuffle=False)

    def _make_dloader(self, split, shuffle=False):
        dataset = self.datasetCls(**self.dataset_kwargs, split=split)
        if len(dataset) == 0:
            return None

        use_persistent_workers = False
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=use_persistent_workers,
            prefetch_factor=2 if self.workers > 0 else None,
            worker_init_fn=self.worker_init_fn if self.workers > 0 else None,
            generator=self.generator if shuffle else None,  # Ensure shuffle is reproducible
        )

    @classmethod
    def add_cli(cls, parser):
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument(
            "--workers",
            type=int,
            default=4,
            help="number of parallel workers for pytorch dataloader",
        )

    def add_dl(self, test_data, batch_size=None, **kwargs):
        # If already a DataLoader, return directly
        if isinstance(test_data, DataLoader):
            return test_data
        
        # Compatible with ray.train.torch (optional dependency)
        try:
            from ray.train.torch import _WrappedDataLoader
            if isinstance(test_data, _WrappedDataLoader):
                return test_data
        except ImportError:
            pass

        if batch_size is None:
            batch_size = self.batch_size
        if not isinstance(test_data, Dataset):
            test_data = self.train.dataset.new(test_data)

        test_data = self.train.new(test_data, batch_size, **kwargs)
        return test_data




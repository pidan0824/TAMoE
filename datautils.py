import os

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset registry: {name: (subdir, csv_filename, dataset_class)}
_DSET_CONFIG = {
    'ettm1':       ('ETT-small',      'ETTm1.csv',           Dataset_ETT_minute),
    'ettm2':       ('ETT-small',      'ETTm2.csv',           Dataset_ETT_minute),
    'etth1':       ('ETT-small',      'ETTh1.csv',           Dataset_ETT_hour),
    'etth2':       ('ETT-small',      'ETTh2.csv',           Dataset_ETT_hour),
    'electricity': ('electricity',    'electricity.csv',      Dataset_Custom),
    'traffic':     ('traffic',        'traffic.csv',          Dataset_Custom),
    'weather':     ('weather',        'weather.csv',          Dataset_Custom),
    'illness':     ('illness',        'national_illness.csv', Dataset_Custom),
    'exchange':    ('exchange_rate',  'exchange_rate.csv',    Dataset_Custom),
}

DSETS = list(_DSET_CONFIG.keys())


def get_dls(params):

    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params, 'use_time_features'): params.use_time_features = False

    seed = getattr(params, 'seed', None)

    subdir, data_path, datasetCls = _DSET_CONFIG[params.dset]
    root_path = os.path.join(_BASE_DIR, 'dataset', subdir)
    size = [params.context_points, 0, params.target_points]

    dls = DataLoaders(
            datasetCls=datasetCls,
            dataset_kwargs={
            'root_path': root_path,
            'data_path': data_path,
            'features': params.features,
            'scale': True,
            'size': size,
            'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
            seed=seed,
            )

    # dataset is assumed to have dimension len x nvars
    sample = dls.train.dataset[0]
    dls.vars = sample[0].shape[1]

    patch_len = getattr(params, 'patch_len', 12)
    stride = getattr(params, 'stride', patch_len)
    assert params.context_points >= patch_len, f"context_points ({params.context_points}) must be >= patch_len ({patch_len})"
    dls.len = (params.context_points - patch_len) // stride + 1
    dls.c = sample[1].shape[0]
    return dls

if __name__ == "__main__":
    class Params:
        dset= 'etth1'
        context_points= 512
        target_points= 96
        batch_size= 128
        num_workers= 4
        features='M'
    params = Params()
    dls = get_dls(params)
    for i, batch in enumerate(dls.valid):
        print(i, len(batch), batch[0].shape, batch[1].shape)

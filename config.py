from PDMD import get_config

CONFIGS = {
    'data_config': {
        'main_path': './',
        'dataset': 'ENERGY_DATASET',
        'model': 'ChemGNN_energy',
    },
    'training_config': {
        'device_type': 'cpu',
        'loss_fn_id': 1,
        'epoch': 2000,
        'epoch_step': 1,
        'batch_size': 1024,
        'lr': 0.002,
        'seed': 0,
        'train_length': 0.8,
        'val_length': 0.2,
    }
}

config = get_config(CONFIGS)

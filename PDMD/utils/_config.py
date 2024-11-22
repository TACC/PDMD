import os

class Config:
    def __init__(self, base_config):
        self.main_path = base_config['data_config']['main_path']
        self.dataset = base_config['data_config']['dataset']
        self.model = base_config['data_config']['model']
        self.device_type = base_config['training_config']['device_type']
        self.loss_fn_id = base_config['training_config']['loss_fn_id']
        self.epoch = base_config['training_config']['epoch']
        self.epoch_step = base_config['training_config']['epoch_step']
        self.batch_size = base_config['training_config']['batch_size']
        self.lr = base_config['training_config']['lr']
        self.seed = base_config['training_config']['seed']
        self.train_ratio = base_config['training_config']['train_length']
        self.val_ratio = base_config['training_config']['val_length']
        self.process_dst_path = os.path.join(self.main_path, 'PDMD_DATASET', self.dataset)
        self.timestring = None


def get_config(base_config):
    model_config = Config(base_config)
    return model_config

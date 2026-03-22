import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset

class MutilWaterDataset(InMemoryDataset):
    def __init__(self, root, split='1water_energy', transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter, log=False)

        path = osp.join(self.processed_dir, f'{self.split}.pt')
        if not osp.exists(path):
            raise FileNotFoundError(f"Processed dataset file not found: {path}")

        self.data, self.slices = torch.load(path, weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return self.root

    @property
    def processed_file_names(self):
        return [f'{self.split}.pt']

    def _download(self):
        return

    def _process(self):
        return

    def download(self):
        return

    def process(self):
        return

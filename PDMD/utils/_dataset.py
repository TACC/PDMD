import torch
import os.path as osp
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)

class MutilWaterDataset(InMemoryDataset):
    def __init__(self, root, split='1water_energy', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path,weights_only=False)

    @property
    def raw_file_names(self):
        return ['train.pickle.npy']

    @property
    def processed_dir(self):
        return self.root

    @property
    def processed_file_names(self):
        return ['1water_energy.pt', '2water_energ.pt', '3water_energ.pt', '4water_energ.pt', '5water_energ.pt',
                '6water_energ.pt', '7water_energ.pt', '8water_energ.pt', '9water_energ.pt', '10water_energ.pt',
                '11water_energ.pt', '12water_energ.pt', '13water_energ.pt', '14water_energ.pt', '15water_energ.pt',
                '16water_energ.pt', '17water_energ.pt', '18water_energ.pt', '19water_energ.pt', '20water_energ.pt',
                '21-water_energ.pt', 'water_energy_optimized.pt']

    def download(self):
        pass

    def process(self):
        pass

import torch
from torch.utils.data import Dataset

class ModDataset(Dataset):
    def __init__(self, mod):
        self.X = torch.tensor([[1, 2, 3],
                               [4, 5, 6],
                               [0, 1, 2],
                               [3, 3, 3],
                               [1, 1, 1]], dtype=torch.int64)
        self.Y = torch.tensor([[0, 1],
                               [1, 2],
                               [2, 3],
                               [3, 4],
                               [4, 5]], dtype=torch.int64) % mod

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def load_dataset(mod):
    return ModDataset(mod)

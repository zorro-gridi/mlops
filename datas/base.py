from torch.utils.data import (
    Dataset,
    )



class Custom_UnitVars_Datasets(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx, :-1]
        y = self.data[idx, -1]
        return X, y


class Custom_MultiVars_Datasets(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx][:-1, :]
        y = self.data[idx][-1, -1]
        return X, y

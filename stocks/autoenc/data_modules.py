import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, random_split


##### DEFINE CUSTOM DATASET CLASS #####
class CallPriceData(Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, mode="train"):
        "Initialization"
        self.df = pd.read_csv("./calldf.csv")
        self.rawdata = self.df.to_numpy()
        self.scaler = MinMaxScaler()
        self.data = self.scaler.fit_transform(self.rawdata)

    def __len__(self):
        "Denotes the total number of samples"
        return self.df.shape[0]

    def __getitem__(self, index):
        "Generates one sample of data"

        # Load data and get label
        X = torch.from_numpy(self.data[index, :-1])
        y = torch.from_numpy(self.data[index, -1:])

        return X.type(torch.float32), y.type(torch.float32)


##### PYTORCH LIGHTNING DATAMODULE? #####
class CallDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.fulldata = CallPriceData()
        self.train_split, self.test_split, self.val_split = random_split(
            self.fulldata, [0.8, 0.1, 0.1], torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_split, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_split, batch_size=self.batch_size)


if __name__ == "__main__":
    call_data = CallDataModule(batch_size=1)
    for idx, (X, y) in enumerate(call_data.train_dataloader()):
        print(idx, y)
        breakpoint()

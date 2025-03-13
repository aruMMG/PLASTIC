import torch
import torch.utils
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import glob
import os

import torch.utils.data

from utils.utils import find_target_dim

def create_dataloader(config, dir, class_names, bs, FC=False, shuffle=True, drop_last=False):
    """
    Creates a dataloader for the Spectroscopy dataset.
    :param dir: Directory containing .npy files
    :param bs: Batch size
    :param shuffle: Whether to shuffle the data
    :param drop_last: Whether to drop the last incomplete batch
    :param inference_mode: If True, returns dataloader for inference
    """
    dataset = SpectroscopyDataset(config, dir, class_names, FC=FC)
    return torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=shuffle, drop_last=drop_last), dataset.config

class SpectroscopyDataset(Dataset):

    def __init__(self, config, dir, class_names, FC=False, raw=False):
        """
        Dataset for Spectroscopy data.
        :param dir: Directory containing .npy files
        :param FC: If True, keeps data in fully connected format
        :param raw: If True, returns raw data
        """
        self.X, self.Y = self.read_data(dir, class_names)
        self.target_len, self.config = find_target_dim(self.X.shape[1], config)
        self.FC = FC
        self.raw = raw

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index):
        X = torch.tensor(self.X[index, :])

        if X.shape[0] != self.target_len:
            X = torch.nn.functional.interpolate(
                X.unsqueeze(0).unsqueeze(0), size=self.target_len, mode='linear', align_corners=False
            ).squeeze()

        if not self.FC:
            X = X.unsqueeze(dim=0)  # Add channel dimension for CNN models
        y = torch.tensor(self.Y[index])
        return X.float(), y.long()

    def read_data(self, dir, class_names):
        X, Y = None, None
        class_label = 0  # Starting label for the first class

        for class_name in class_names:
            file = os.path.join(dir, class_name + ".npy")
            data = np.load(file)
            print(f'read {file} and shape is: {data.shape}')

            if X is None:
                X = data
                Y = np.full(data.shape[0], class_label, dtype=int)
            else:
                X = np.concatenate([X, data])
                Y = np.concatenate([Y, np.full(data.shape[0], class_label, dtype=int)])

            class_label += 1  # Increment the class label for the next file

        return X, Y


class InferenceDataset(Dataset):
    """
    Dataset class for inference, handling only features (X).
    """

    def __init__(self, dir, FC=False, raw=False):
        """
        :param dir: Directory containing .npy files
        :param FC: If True, keeps data in fully connected format
        :param raw: If True, returns raw data
        """
        self.X = self.read_data(dir)
        self.FC = FC
        self.raw = raw

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.FC:
            X = torch.tensor(self.X[index, :])
        else:
            X = torch.tensor(self.X[index, :]).unsqueeze(dim=0)
        return X.float()

    def read_data(self, dir):
        """
        Reads data from directory containing .npy files.
        """
        X = None

        for file in glob.glob(os.path.join(dir, "*.npy")):
            data = np.load(file)
            print(f"Read {file} and shape is: {data.shape}")

            if X is None:
                X = data
            else:
                X = np.concatenate([X, data])

        return X


def create_inference_dataloader(dir, bs, FC=False, shuffle=False, drop_last=False):
    """
    Creates a dataloader for inference.
    :param dir: Directory containing .npy files
    :param bs: Batch size
    :param FC: If True, keeps data in fully connected format
    :param shuffle: Whether to shuffle the data
    :param drop_last: Whether to drop the last incomplete batch
    """
    dataset = InferenceDataset(dir, FC=FC)
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle, drop_last=drop_last)


if __name__=="__main__":
        dir = ""
        dataset = SpectroscopyDataset(dir, FC=True)
        dataloader = DataLoader(dataset, batch_size=2,shuffle=True)
        for data, labels, data_num in dataloader:
            print(data.dtype)
            print(labels.dtype)

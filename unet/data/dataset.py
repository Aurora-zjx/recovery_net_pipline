# -*- coding: utf-8 -*-
from torch.utils import data
import numpy as np
import h5py

class DatasetFromFolder(data.Dataset):
    # def __init__(self, train_set_path, label_set_path, drive_set_path, d_drive_set_path):
    def __init__(self, train_set_path, label_set_path):
        super(DatasetFromFolder, self).__init__()
        self.train_set_path = train_set_path
        self.label_set_path = label_set_path
        # self.drive_set_path = drive_set_path
        # self.d_drive_set_path = d_drive_set_path
    def __getitem__(self, index):
        train_data = np.loadtxt(self.train_set_path)
        label_data = np.loadtxt(self.label_set_path)
        # drive_data = np.loadtxt(self.drive_set_path)
        # d_drive_set_path = np.loadtxt(self.d_drive_set_path)
        # return train_data[index, :], label_data[index, :], drive_data[index, :], d_drive_set_path[index, :]
        return train_data[index, :], label_data[index, :]

    def __len__(self):
        return len(np.loadtxt(self.train_set_path))


class DatasetFromH5DY(data.Dataset):
    def __init__(self, h5_file):
        super(DatasetFromH5DY, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return f['ori'][idx],f['label'][idx]
            # return np.expand_dims(f['ori'][idx] / 255., 0), np.expand_dims(f['label'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['ori'])


class DatasetFromFolder_test(data.Dataset):
    def __init__(self, train_set_path, label_set_path):
        super(DatasetFromFolder_test, self).__init__()
        self.train_set_path = train_set_path
        self.label_set_path = label_set_path
        
    def __getitem__(self, index):
        train_data = np.loadtxt(self.train_set_path)
        label_data = np.loadtxt(self.label_set_path)
        
        return train_data[index, :], label_data[index, :]

    def __len__(self):
        return len(np.loadtxt(self.train_set_path))


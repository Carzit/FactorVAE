import os
import random
from numbers import Number
from typing import List, Dict, Tuple, Optional, Literal, Union, Any

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class StockDataset(Dataset):
    def __init__(self, data_x_dir:str, data_y_dir:str, label_name:str) -> None:
        super().__init__()
        self.data_x_dir:str = data_x_dir
        self.data_y_dir:str = data_y_dir

        self.label_name = label_name

        self.x_file_paths:List[str] = [os.path.join(data_x_dir, f) for f in os.listdir(data_x_dir) if f.endswith('.csv')]
        self.y_file_paths:List[str] = [os.path.join(data_y_dir, f) for f in os.listdir(data_y_dir) if f.endswith('.csv')]

    def __getitem__(self, index) -> Tuple[torch.Tensor]:
        X = pd.read_csv(self.x_file_paths[index], index_col=[0,1,2])
        X = torch.from_numpy(X.values).float()
        y = pd.read_csv(self.y_file_paths[index], index_col=[0,1,2])[self.label_name]
        y = torch.from_numpy(y.values).float()
        return X, y
    
    def __len__(self):
        return min(len(self.x_file_paths), len(self.y_file_paths))
    
    def info(self):
        x, y = self[0]
        x_shape = x.shape
        y_shape = y.shape
        return {"x_shape":x_shape, "y_shape":y_shape, "len":len(self)}
    
    def serial_split(self, ratios:List[Number]) -> List["StockDataset"]:
        total_length = len(self)
        split_lengths = list(map(lambda x:round(x / sum(ratios) * total_length), ratios))
        split_lengths[0] = total_length - sum(split_lengths[1:])
        splitted_datasets = []

        i = 0
        for j in split_lengths:
            splitted_dataset = StockDataset(data_x_dir=self.data_x_dir, data_y_dir=self.data_y_dir, label_name=self.label_name)
            splitted_dataset.x_file_paths = splitted_dataset.x_file_paths[i:i+j]
            splitted_dataset.y_file_paths = splitted_dataset.y_file_paths[i:i+j]
            splitted_datasets.append(splitted_dataset)
            i += j

        return splitted_datasets

    
    def random_split(self, ratios:List[Number]) -> List["StockDataset"]:
        total_length = len(self)
        split_lengths = list(map(lambda x:round(x / sum(ratios) * total_length), ratios))
        split_lengths[0] = total_length - sum(split_lengths[1:])
        splitted_datasets = []

        base_names = [os.path.basename(file_path) for file_path in self.x_file_paths]
        random.shuffle(base_names)
        shuffled_x_file_paths = [os.path.join(self.data_x_dir, base_name) for base_name in base_names]
        shuffled_y_file_paths = [os.path.join(self.data_y_dir, base_name) for base_name in base_names]

        i = 0
        for j in split_lengths:
            splitted_dataset = StockDataset(data_x_dir=self.data_x_dir, data_y_dir=self.data_y_dir, label_name=self.label_name)
            splitted_dataset.x_file_paths = shuffled_x_file_paths[i:i+j]
            splitted_dataset.y_file_paths = shuffled_y_file_paths[i:i+j]
            splitted_datasets.append(splitted_dataset)
            i += j
            
        return splitted_datasets

class StockSequenceDataset(Dataset):
    def __init__(self, stock_dataset:StockDataset, seq_len:int) -> None:
        super().__init__()
        self.stock_dataset = stock_dataset
        self.seq_len = seq_len

    def __len__(self):
        return len(self.stock_dataset) - self.seq_len + 1
    
    def __getitem__(self, index) -> Tuple[torch.Tensor]:
        X_seq = torch.stack([self.stock_dataset[i][0] for i in range(index, index+self.seq_len)], dim=0)
        y = self.stock_dataset[index+self.seq_len-1][1]
        return X_seq, y
    
if __name__ == "__main__":
    dataset = StockDataset(data_x_dir=r"C:\Users\C'heng\PycharmProjects\SWHY\data\demo\preprocess\alpha",
                           data_y_dir=r"C:\Users\C'heng\PycharmProjects\SWHY\data\demo\preprocess\label",
                           label_name="ret10")
    dataset_seq = StockSequenceDataset(dataset, 10)
    print(dataset_seq[1])






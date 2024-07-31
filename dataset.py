import os
import sys
import random
import datetime
import logging
import argparse
from numbers import Number
from typing import List, Dict, Tuple, Optional, Literal, Union, Any, Callable

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

def parse_args():
    parser = argparse.ArgumentParser(description="Data acquisition and dataset generation.")

    parser.add_argument("--log_folder", type=str, default=os.curdir, help="Path of folder for log file. Default `.`")
    parser.add_argument("--log_name", type=str, default="log.txt", help="Name of log file. Default `log.txt`")

    parser.add_argument("--x_folder", type=str, required=True, help="Path of folder for x (alpha) data csv files")
    parser.add_argument("--y_folder", type=str, required=True, help="Path of folder for y (label) data csv files")
    parser.add_argument("--label_name", type=str, required=True, help="Target label name (col name in y files)")

    parser.add_argument("--split_ratio", type=float, nargs=3, default=[0.7, 0.2, 0.1], help="Split ratio for train-validation-test. Default 0.7, 0.2, 0.1")
    parser.add_argument("--train_seq_len", type=int, required=True, help="Sequence length (num of days) for train dataset")
    parser.add_argument("--val_seq_len", type=int, default=None, help="Sequence length (num of days) for validation dataset. If not specified, default equal to train_seq_len.")
    parser.add_argument("--test_seq_len", type=int, default=None, help="Sequence length (num of days) for test dataset. If not specified, default equal to train_seq_len.")

    parser.add_argument("--save_path", type=str, required=True, help="Path to save the dataset dictionary.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    log_folder = args.log_folder
    log_name = args.log_name
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - [%(levelname)s] : %(message)s',
        handlers=[logging.FileHandler(os.path.join(log_folder, log_name)), logging.StreamHandler()])
    
    logging.debug(f"Command: {' '.join(sys.argv)}")
    logging.debug(f"Params: {vars(args)}")

    dataset = StockDataset(data_x_dir=args.x_folder,
                           data_y_dir=args.y_folder,
                           label_name=args.label_name)
    train_set, val_set, test_set = dataset.serial_split(args.split_ratio)
    train_set = StockSequenceDataset(train_set, seq_len=args.train_seq_len)
    val_set = StockSequenceDataset(val_set, seq_len=args.val_seq_len or args.train_seq_len)
    test_set = StockSequenceDataset(val_set, seq_len=args.test_seq_len or args.train_seq_len)

    torch.save({"train": train_set, "val": val_set, "test": test_set}, args.save_path)

    # python dataset.py --x_folder "C:\Users\C'heng\PycharmProjects\SWHY\data\test\alpha" --y_folder "C:\Users\C'heng\PycharmProjects\SWHY\data\test\label" --label_name "ret10" --train_seq_len 2 --save_path "dataset.pt"






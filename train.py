import os
import sys
import datetime
import logging
import argparse
from numbers import Number
from typing import List, Dict, Tuple, Optional, Literal, Union, Any, Callable

from tqdm import tqdm
from safetensors.torch import save_file, load_file

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from dataset import StockDataset, StockSequenceDataset
from model import FactorVAE
from loss import ObjectiveLoss

class FactorVAETrainer:
    def __init__(self,
                 model:FactorVAE,
                 loss_func:ObjectiveLoss,
                 optimizer:torch.optim.Optimizer,
                 lr_scheduler:Optional[torch.optim.lr_scheduler.LRScheduler] = None,
                 device:torch.device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) -> None:
        
        self.model:nn.Module = model
        self.loss_func:nn.Module = loss_func
        self.optimizer:nn.Module = optimizer
        self.lr_scheduler:Optional[torch.optim.lr_scheduler.LRScheduler] = lr_scheduler

        self.train_loader:DataLoader
        self.val_loader:DataLoader

        self.writer:SummaryWriter = None

        self.max_epoches:int
        self.hparams:Optional[dict]

        self.log_folder:str = "log"
        self.sample_batch:int = 0
        self.report_epoch:int = 1
        self.save_epoch:int = 1
        self.save_folder:str = os.curdir
        self.save_name:str = "Model"
        self.save_format:Literal[".pt", ".safetensors"] = ".pt"

        self.device = device
        
    def load_dataset(self, 
                     train_set:StockSequenceDataset, 
                     val_set:StockSequenceDataset,
                     shuffle:bool = True,
                     batch_size:Optional[int] = None):
        self.train_loader = DataLoader(dataset=train_set,
                                          batch_size=batch_size, 
                                          shuffle=shuffle)
        self.val_loader = DataLoader(dataset=val_set, 
                                     batch_size=batch_size,
                                     shuffle=shuffle)
        
    def save_checkpoint(self, 
                        save_folder:str, 
                        save_name:str, 
                        save_format:Literal[".pt",".safetensors"]=".pt"):
        save_path = os.path.join(save_folder, save_name+save_format)
        if save_format == ".pt":
            torch.save(self.model.state_dict(), save_path)
        elif save_format == ".safetensors":
            save_file(self.model.state_dict(), save_path)

    def load_checkpoint(self,
                        model_path:str):
        if model_path.endswith(".pt"):
            self.model.load_state_dict(torch.load(model_path))
        elif model_path.endswith(".safetensors"):
            self.model.load_state_dict(load_file(model_path))

    def set_configs(self,
                    max_epoches:int,
                    hparams:Optional[dict] = None,
                    log_folder:str = "log",
                    sample_batch:int = 0,
                    report_epoch:int=1,
                    save_epoch:int=1,
                    save_forder:str=os.curdir,
                    save_name:str="Model",
                    save_format:str=".pt"):
        
        self.max_epoches = max_epoches
        self.hparams = hparams
        
        self.log_folder = log_folder
        self.sample_batch = sample_batch
        self.report_epoch = report_epoch
        self.save_epoch = save_epoch
        self.save_folder = save_forder
        self.save_name = save_name
        self.save_format = save_format

        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self.writer = SummaryWriter(
            os.path.join(
                self.log_folder, f"TRAIN_{self.save_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            ))
        
    def train(self):
        writer = self.writer
        model = self.model.to(device=self.device)
        loss_func = self.loss_func
        optimizer = self.optimizer

        # Train
        for epoch in range(self.max_epoches):
            train_loss_list = []
            val_loss_list = []
            
            # Train one Epoch
            model.train()
            for batch, (X, y) in enumerate(tqdm(self.train_loader)):
                optimizer.zero_grad()

                X = X.to(device=self.device)
                y = y.to(device=self.device)

                y_hat, mu_posterior, sigma_posterior, mu_prior, sigma_prior = model(X, y)
                train_loss = loss_func(y, y_hat, mu_prior, sigma_prior, mu_posterior, sigma_posterior)
                
                train_loss.backward()
                optimizer.step()
                train_loss_list.append(train_loss.item())
                
                if self.sample_batch:
                    if (batch+1) % self.sample_batch == 0:
                        print(f"\n[Batch {batch+1}] \nX:{X} \ny:{y} \nloss:{train_loss.item()} \ny_hat:{y_hat} \nmu_prior:{mu_prior} \nsigma_prior:{sigma_prior} \nmu_posterior:{mu_posterior} \nsigma_posterior:{sigma_posterior}")
              
            # Record Train Loss Scalar
            train_loss_epoch = sum(train_loss_list)/len(train_loss_list)
            writer.add_scalar("Train Loss", train_loss_epoch, epoch+1)
            
            # Calculate val loss without recording grad.

            model.eval() # set eval mode to frozen layers like dropout
            with torch.no_grad(): 
                for batch, (X, y) in enumerate(self.val_loader):
                    X = X.to(device=self.device)
                    y = y.to(device=self.device)
                    y_hat, mu_posterior, sigma_posterior, mu_prior, sigma_prior = model(X, y)
                    val_loss = loss_func(y, y_hat, mu_prior, sigma_prior, mu_posterior, sigma_posterior)
                    val_loss_list.append(val_loss.item())

                val_loss_epoch = sum(val_loss_list) / len(val_loss_list)  
                writer.add_scalar("Validation Loss", val_loss_epoch, epoch+1)
                writer.add_scalars("Train-Val Loss", {"Train Loss": train_loss_epoch, "Validation Loss": val_loss_epoch}, epoch+1)

            # If hyper parameters passed, record it in hparams.
            if self.hparams:
                writer.add_hparams(hparam_dict=self.hparams, metric_dict={"hparam/TrainLoss":train_loss_epoch, "hparam/ValLoss":val_loss_epoch})

            # If learning rate scheduler exisit, update learning rate per epoch.
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], epoch+1)
            if self.lr_scheduler:
                self.lr_scheduler.step()
            
            # Flushes the event file to disk
            writer.flush()

            # Specify print_per_epoch = 0 to unable print training information.
            if self.report_epoch:
                if (epoch+1) % self.report_epoch == 0:
                    print('Epoch [{}/{}], Train Loss: {:.6f}, Validation Loss: {:.6f}'.format(epoch+1, self.max_epoches, train_loss_epoch, val_loss_epoch))
            
            # Specify save_per_epoch = 0 to unable save model. Only the final model will be saved.
            if self.save_epoch:
                if (epoch+1) % self.save_epoch == 0:
                    model_name = f"{self.save_name}_epoch{epoch+1}"
                    self.save_checkpoint(save_folder=self.save_folder,
                                         save_name=model_name,
                                         save_format=self.save_format)

        writer.close()


def parse_args():
    # TODO
    parser = argparse.ArgumentParser(description="Distributed Data Normalizer.")

    parser.add_argument("--log_folder", type=str, default=os.curdir, help="Path of folder for log file. Default `.`")
    parser.add_argument("--log_name", type=str, default="log.txt", help="Name of log file. Default `log.txt`")

    parser.add_argument("--data_folder", type=str, required=True, help="Path of folder for csv files")
    parser.add_argument("--save_folder", type=str, default=None, help="Path of folder for Normalizer to save processed result. If not specified, files in data folder will be replaced.")
    parser.add_argument("--mode", type=str, default="cs_zscore", help="Normalization mode, literally `cs_zscore`, `cs_rank`, `global_zscore`, `global_minmax` or `global_robust_zscore`. Default `cs_score`.")

    return parser.parse_args()


if __name__ == "__main__":
    dataset = StockDataset(data_x_dir=r"D:\PycharmProjects\SWHY\data\demo\alpha",
                           data_y_dir=r"D:\PycharmProjects\SWHY\data\demo\label",
                           label_name="ret10")
    train_set, val_set = dataset.serial_split([0.7, 0.3])
    train_set = StockSequenceDataset(train_set, seq_len=5)
    val_set = StockSequenceDataset(val_set, 5)

    model = FactorVAE(input_size=101, num_gru_layers=4, gru_hidden_size=32, hidden_size=16, latent_size=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = ObjectiveLoss(gamma=0.8)

    trainer = FactorVAETrainer(model=model,
                               loss_func=loss_func,
                               optimizer=optimizer)
    trainer.load_dataset(train_set=train_set, val_set=val_set)
    
    trainer.set_configs(max_epoches=20, sample_batch=200)
    print("start training...")
    trainer.train()



import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from dataset import StockDataset, StockSequenceDataset
from nets import FactorVAE
from loss import ObjectiveLoss
from train import FactorVAETrainer

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == "__main__":
    datasets = torch.load(r"D:\PycharmProjects\SWHY\data\preprocess\dataset_cs_zscore.pt")
    test_set = datasets["val"]

    model = FactorVAE(input_size=101, 
                      num_gru_layers=2, 
                      gru_hidden_size=32, 
                      hidden_size=16, 
                      latent_size=4,
                      gru_drop_out=0.1)
    
    trainer = FactorVAETrainer(model=model)
    trainer.load_checkpoint(r"D:\PycharmProjects\SWHY\model\factor-vae\model1\model3_epoch20.pt")
    #print(trainer.model.feature_extractor.state_dict)
    #print(trainer.eval(test_set, "MSE"))
    
    x, y = test_set[48]
    print(x,y,x.amax(), y.amax())
    y_pred = trainer.model.predict(x)
    print(y,y_pred)

#Imports

import torch
import numpy as np
import os
import json
import joblib
from torch.utils.data import Dataset,DataLoader
from itertools import repeat
import pandas as pd
import math
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import wandb
import logging
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os
from pathlib import Path

data_dir =Path('/common/home/vk405/Projects/EnergyLab/Solar-forcast/Data/')
image_dir = Path('/common/users/vk405/EnergyLab/Data')
proc_data_dir = Path('/common/users/vk405/EnergyLab/Data/ProcData')

logger = logging.getLogger(__name__)
wandb_logger = lambda dir, version: WandbLogger(
    name="wandb", save_dir=dir, version=version
)
csvlogger = lambda dir, version: CSVLogger(dir, name="csvlogs", version=version)
tblogger = lambda dir, version: TensorBoardLogger(dir, name="tblogs", version=version)

def get_loggers(dir,version,lis=["csv"]):
    lgrs = []
    if "wandb" in lis:
        lgrs.append(wandb_logger(dir, version))
    if "csv" in lis:
        lgrs.append(csvlogger(dir, version))
    if "tb" in lis:
        lgrs.append(tblogger(dir, version))
    return lgrs



# Model

def SCNN(n_ch=6):
    
    model = []
    model.append(nn.Conv2d(in_channels=n_ch, out_channels=64, kernel_size=(3, 3), padding="same"))
    model.append(nn.ReLU())
    model.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same"))
    model.append(nn.ReLU())
    model.append(nn.MaxPool2d((2,2),stride=(2,2)))
    
    model.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same"))
    model.append(nn.ReLU())
    model.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same"))
    model.append(nn.ReLU())
    model.append(nn.MaxPool2d((2,2),stride=(2,2)))

    model.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding="same"))
    model.append(nn.ReLU())
    model.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding="same"))
    model.append(nn.ReLU())
    model.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding="same"))
    model.append(nn.ReLU())
    model.append(nn.MaxPool2d((2,2),stride=(2,2)))

    model.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding="same"))
    model.append(nn.ReLU())
    model.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding="same"))
    model.append(nn.ReLU())
    #model.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding="same"))
    #model.append(nn.ReLU())
    model.append(nn.MaxPool2d((2,2),stride=(2,2)))

    model.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding="same"))
    model.append(nn.ReLU())
    model.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding="same"))
    model.append(nn.ReLU())
    #model.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding="same"))
    #model.append(nn.ReLU())
    model.append(nn.MaxPool2d((2,2),stride=(2,2)))

    model.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding="same"))
    model.append(nn.ReLU())
    model.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding="same"))
    model.append(nn.ReLU())
    model.append(nn.MaxPool2d((2,2),stride=(2,2)))




    return nn.Sequential(*model)


class VggModel(pl.LightningModule):
    def __init__(self,hparams,dset=None):
        super().__init__()
        self.save_hyperparameters()

        
        if hparams.framecnt != 2:
            raise NotImplementedError

        #get the complete model
        base_model = [SCNN(hparams.framecnt*3)]
        base_model.extend([nn.Flatten(),nn.Dropout(hparams.dropoutp),nn.Linear(8192,256),nn.Dropout(hparams.dropoutp),nn.Linear(256,1)])
        self.net = nn.Sequential(*base_model)

    def forward(self,x):
        #keep this for inference
        out = self.net(x)
        return out

    
    def training_step(self,batch,batch_idx):
        #for training
        x,y = batch
        y_hat = torch.squeeze(self.net(x.float()))

        loss = F.l1_loss(y_hat,y.float())
        self.log("train_loss",loss,on_step=True)
        return loss

    
    def validation_step(self,batch,batch_idx):
        #for training
        x,y = batch
        y_hat = torch.squeeze(self.net(x.float()))

        loss = F.l1_loss(y_hat,y.float())
        self.log("val_loss",loss,on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        lr = self.hparams.lr if 'lr' in self.hparams else 1e-5
        #lr=0.00001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False
        optimizer = torch.optim.Adam(self.parameters(), lr=lr,betas=(0.9,0.999),amsgrad=False)
        return optimizer



# DataSet

#dset

class Dset(Dataset):
    def __init__(self,split= 'train',data_dir='/common/home/vk405/Projects/EnergyLab/Solar-forcast/Data/',
    image_dir='/common/users/vk405/EnergyLab/Data/ProcData'):
        self.split = split

        #hardcoded dir locs 
        self.data_dir = Path(data_dir)
        self.image_dir = Path(image_dir) 
        
        base_df = pd.read_csv(f'{self.data_dir}/tgtimgs.csv')
        thresh = base_df['10ma_tgt'].quantile(0.99)
        base_df.loc[base_df.index[base_df['10ma_tgt']>=thresh],'10ma_tgt'] = thresh
        
        trn_df = pd.DataFrame(base_df.loc[pd.to_datetime(base_df['Date']).dt.year <= 2014])
        self.scalar = MinMaxScaler().fit(trn_df['10ma_tgt'].values.reshape(-1,1))
        #splits are hardcoded as per the original paper
        if self.split == 'train':
            self.df = pd.DataFrame(base_df.loc[pd.to_datetime(base_df['Date']).dt.year <= 2014])
            
        elif self.split == 'valid':
            self.df = pd.DataFrame(base_df.loc[pd.to_datetime(base_df['Date']).dt.year == 2015])
            
        else:
            self.df = pd.DataFrame(base_df.loc[pd.to_datetime(base_df['Date']).dt.year > 2015])
        #set an upper-threshold.
        self.df.loc[self.df.index[self.df['10ma_tgt']>=thresh],'10ma_tgt'] = thresh
        #self.df[self.df['Target']>=1300.0]['Target'] = 1300.0
        rescaled = self.scalar.transform(self.df['Target'].values.reshape(-1,1))
        self.df['10ma_tgt'] = np.squeeze(rescaled)
        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self,ind):
        try:
            imgs = sorted(eval(self.df.iloc[ind]['Imgs']),key = lambda x: int(x.split('/')[-1].split('.')[0]))
            target = self.df.iloc[ind]['10ma_tgt']
            #image_dir = Path('/common/users/vk405/EnergyLab/Data/ProcData') 
            proc_imgs = [self.image_dir/(x.split('/')[-1].split('.')[0]+'.joblib') for x in imgs]
            proc_arrays = [joblib.load(ele) for ele in proc_imgs]
        except:
            print(ind)
            
        return (np.concatenate(proc_arrays,-1).reshape(-1,256,256))/255.0,target






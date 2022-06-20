from cProfile import run
import numpy as np
import pandas as pd
import os
from pathlib import Path
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pandas
from tqdm import tqdm
import joblib

import torch
import os
import json
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from sklearn.preprocessing import MinMaxScaler

import wandb
import logging
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import os

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




data_dir =Path('/common/home/vk405/Projects/EnergyLab/Solar-forcast/Data/')
image_dir = Path('/common/users/vk405/EnergyLab/Data')

class Dset(Dataset):
    def __init__(self,split= 'train',data_dir='/common/home/vk405/Projects/EnergyLab/Solar-forcast/Data/',
    image_dir='/common/users/vk405/EnergyLab/Data/ProcData'):
        self.split = split

        #hardcoded dir locs 
        self.data_dir = Path(data_dir)
        self.image_dir = Path(image_dir) 
        
        base_df = pd.read_csv(f'{self.data_dir}/tgtimgs.csv')
        base_df.loc[base_df.index[base_df['Target']>=1250.0],'Target'] = 1250.0
        
        trn_df = pd.DataFrame(base_df.loc[pd.to_datetime(base_df['Date']).dt.year <= 2014])
        self.scalar = MinMaxScaler().fit(trn_df['Target'].values.reshape(-1,1))
        #splits are hardcoded as per the original paper
        if self.split == 'train':
            self.df = pd.DataFrame(base_df.loc[pd.to_datetime(base_df['Date']).dt.year <= 2014])
            
        elif self.split == 'valid':
            self.df = pd.DataFrame(base_df.loc[pd.to_datetime(base_df['Date']).dt.year == 2015])
            
        else:
            self.df = pd.DataFrame(base_df.loc[pd.to_datetime(base_df['Date']).dt.year > 2015])
        #set an upper-threshold.
        self.df.loc[self.df.index[self.df['Target']>=1250.0],'Target'] = 1250.0
        #self.df[self.df['Target']>=1300.0]['Target'] = 1300.0
        rescaled = self.scalar.transform(self.df['Target'].values.reshape(-1,1))
        self.df['Target'] = np.squeeze(rescaled)
        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self,ind):
        try:
            imgs = sorted(eval(self.df.iloc[ind]['Imgs']),key = lambda x: int(x.split('/')[-1].split('.')[0]))
            target = self.df.iloc[ind]['Target']
            #image_dir = Path('/common/users/vk405/EnergyLab/Data/ProcData') 
            proc_imgs = [self.image_dir/(x.split('/')[-1].split('.')[0]+'.joblib') for x in imgs]
            proc_arrays = [joblib.load(ele) for ele in proc_imgs]
        except:
            print(ind)
            
        return (np.concatenate(proc_arrays,-1).reshape(-1,256,256))/255.0,target




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


class CombModel(pl.LightningModule):
    def __init__(self,hparams,dset=None):
        super().__init__()
        self.save_hyperparameters()

        
        if hparams.framecnt != 2:
            raise NotImplementedError

        #get the complete model
        base_model = [SCNN(hparams.framecnt*3)]
        base_model.extend([nn.Flatten(),nn.Dropout(hparams.dropoutp),nn.Linear(8192,256)])
        self.prenet = nn.Sequential(*base_model)
        #nn.Dropout(hparams.dropoutp),nn.Linear(256,1)
        self.autoregressnet = nn.Sequential(nn.Linear(hparams.lkback,hparams.lkback//2),nn.ReLU())

        self.combined_net = nn.Sequential(nn.Dropout(hparams.dropoutp),nn.Linear(256+hparams.lkback//2,1))

    def forward(self,x,lkbk_vals):
        #keep this for inference
        image_out = self.prenet(x)
        regres_out = self.autoregressnet(lkbk_vals)
        out  = self.combined_net(torch.concat([image_out,regres_out],dim=-1))
        return out

    
    def training_step(self,batch,batch_idx):
        #for training
        x,x_r,y = batch
        y_hat = torch.squeeze(self.forward(x.float(),x_r.float()))

        loss = F.l1_loss(y_hat,y.float())
        self.log("train_loss",loss,on_step=True)
        return loss

    
    def validation_step(self,batch,batch_idx):
        #for training
        x,x_r,y = batch
        y_hat = torch.squeeze(self.forward(x.float(),x_r.float()))


        loss = F.l1_loss(y_hat,y.float())
        self.log("val_loss",loss,on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        lr = self.hparams.lr if 'lr' in self.hparams else 1e-5
        #lr=0.00001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False
        optimizer = torch.optim.Adam(self.parameters(), lr=lr,betas=(0.9,0.999),amsgrad=False)
        return optimizer




#new dataset

class Dset(Dataset):
    def __init__(self,split= 'train',data_dir='/common/home/vk405/Projects/EnergyLab/Solar-forcast/Data/',
    image_dir='/common/users/vk405/EnergyLab/Data/ProcData',lkback=10):
        self.split = split

        #hardcoded dir locs 
        self.data_dir = Path(data_dir)
        self.image_dir = Path(image_dir) 
        self.lkback = lkback
        
        base_df = pd.read_csv(f'{self.data_dir}/tgtimgs.csv')
        base_df.loc[base_df.index[base_df['Target']>=1250.0],'Target'] = 1250.0
        
        trn_df = pd.DataFrame(base_df.loc[pd.to_datetime(base_df['Date']).dt.year <= 2014])
        self.scalar = MinMaxScaler().fit(trn_df['Target'].values.reshape(-1,1))

        trn_msi = joblib.load(self.data_dir /'trn_msi.joblib')
        trn_msi[trn_msi>=1250] = 1250.0
        self.msi_scalar = MinMaxScaler().fit(trn_msi.reshape(-1,1))
        #splits are hardcoded as per the original paper
        if self.split == 'train':
            self.df = pd.DataFrame(base_df.loc[pd.to_datetime(base_df['Date']).dt.year <= 2014])
            self.msi = joblib.load(self.data_dir /'trn_msi.joblib')
        
            
        elif self.split == 'valid':
            self.df = pd.DataFrame(base_df.loc[pd.to_datetime(base_df['Date']).dt.year == 2015])
            self.msi = joblib.load(self.data_dir /'vld_msi.joblib')
            
        else:
            self.df = pd.DataFrame(base_df.loc[pd.to_datetime(base_df['Date']).dt.year > 2015])
            self.msi = joblib.load(self.data_dir /'tst_msi.joblib')
        #set an upper-threshold.
        self.df.loc[self.df.index[self.df['Target']>=1250.0],'Target'] = 1250.0
        #self.df[self.df['Target']>=1300.0]['Target'] = 1300.0
        rescaled = self.scalar.transform(self.df['Target'].values.reshape(-1,1))
        self.df['Target'] = np.squeeze(rescaled)
        self.msi = np.squeeze(self.msi_scalar.transform(self.msi.reshape(-1,1)))
        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self,ind):
        try:
            imgs = sorted(eval(self.df.iloc[ind]['Imgs']),key = lambda x: int(x.split('/')[-1].split('.')[0]))
            target = self.df.iloc[ind]['Target']
            #image_dir = Path('/common/users/vk405/EnergyLab/Data/ProcData') 
            proc_imgs = [self.image_dir/(x.split('/')[-1].split('.')[0]+'.joblib') for x in imgs]
            proc_arrays = [joblib.load(ele) for ele in proc_imgs]
            proc_lkback = self.msi[ind*self.lkback:ind*self.lkback+10]
        except:
            print(ind)
            
        return (np.concatenate(proc_arrays,-1).reshape(-1,256,256))/255.0,proc_lkback,target



def run_model(cfg):
    pl.seed_everything(cfg.seed)
    dir = cfg.artifacts_loc
    version = str(cfg.version)
    logger_list = get_loggers(dir, version,cfg.loggers)
    cbs = []
    if "early_stop" in cfg.cbs:
        #? does'nt really work atm
        #params = cfg.model.cbs.early_stop
        earlystopcb = EarlyStopping(monitor='val_loss',mode="min",patience=3,verbose=False)
        cbs.append(earlystopcb)
    if "checkpoint" in cfg.cbs:
        store_path = dir + "ckpts/" + str(cfg.version) + "/"
        isExist = os.path.exists(store_path)
        if not isExist:
            os.makedirs(store_path)
        fname = "{epoch}-{val_loss:.2f}"
        params = cfg.checkpoint
        checkptcb = ModelCheckpoint(**params, dirpath=store_path, filename=fname)
        cbs.append(checkptcb)

    #wandb.init(project="solarforecast", config=cfg)
    if cfg.mode == 'train':
        trn_fdata = Dset(data_dir=cfg.data_dir,split=cfg.mode,image_dir= cfg.image_dir,\
               lkback=cfg.lkback)
        vld_fdata = Dset(data_dir=cfg.data_dir,split='valid',image_dir= cfg.image_dir,lkback=cfg.lkback)

        val_loader = DataLoader(vld_fdata,\
            batch_size=cfg.batch_size,shuffle=False,num_workers=4,pin_memory=True)
        train_loader = DataLoader(trn_fdata,\
            batch_size=cfg.batch_size,shuffle=True,num_workers=4,pin_memory=True)
            
        hparams = cfg    
        net = CombModel(hparams)
        trainer = pl.Trainer(
            logger=logger_list,callbacks=cbs, gpus=[3],deterministic=True, **cfg.trainer
        )
        trainer.fit(net,train_dataloaders=train_loader,val_dataloaders=val_loader)
        return trainer
        #trainer.tune(net,train_loader)
            
    else:
        pass
    

if __name__=='__main__':
    from argparse import Namespace
    cfg = Namespace(
        version = 'combtrail',
        artifacts_loc = "/common/home/vk405/Projects/EnergyLab/Solar-forcast/artifacts/",
        data_dir = "/common/home/vk405/Projects/EnergyLab/Solar-forcast/Data/",
        image_dir = "/common/users/vk405/EnergyLab/Data/ProcData/",
        mode = 'train',
        loggers = ["csv"],
        seed = 0,
        cbs = ["checkpoint"],
        trainer = {'log_every_n_steps': 50,
        'max_epochs': 40},
        checkpoint = {"save_top_k": 5,
        "monitor": "val_loss","mode":"min"},
        dropoutp=0.2,
        framecnt=2,
        lr = 1.5e-5,
        batch_size=64,
        lkback = 10
    )
    run_model(cfg)
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
from torch.autograd import Variable

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


from pathlib import Path
data_dir =Path('/common/home/vk405/Projects/EnergyLab/Solar-forcast/Data/')
image_dir = Path('/common/users/vk405/EnergyLab/Data')
proc_data_dir = Path('/common/users/vk405/EnergyLab/Data/ProcData')
#df = pd.read_csv(f'{data_dir}/tgtimgs.csv')

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers,seq_length):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out

class Dset(Dataset):
    def __init__(self,split= 'train',data_dir='/common/home/vk405/Projects/EnergyLab/Solar-forcast/Data/',
    image_dir='/common/users/vk405/EnergyLab/Data/ProcData',seq_length=10):
        self.split = split

        #hardcoded dir locs 
        self.data_dir = Path(data_dir)
        self.image_dir = Path(image_dir) 
        self.seq_length = seq_length
        
        base_df = pd.read_csv(f'{self.data_dir}/tgtimgs.csv')
        dt_lis = []
        for i in tqdm(range(len(base_df))):
            dt = ' '.join([base_df.iloc[i]['Date'],base_df.iloc[i]['MST']])
            dt_lis.append(dt)
        base_df['DateTime'] =  dt_lis
        
        df_svd = pd.read_csv(os.path.join(data_dir, 'SRRL_measurement_timeseries.csv'))
        base_df['GHI'] = df_svd[df_svd['DateTime'].isin(set(base_df['DateTime']))].sort_values(['Date','MST'])['GHI'].values
        
        base_df.loc[base_df.index[base_df['GHI']>=1250.0],'GHI'] = 1250.0
        base_df.loc[base_df.index[base_df['Target']>=1250.0],'Target'] = 1250.0

        trn_df = pd.DataFrame(base_df.loc[pd.to_datetime(base_df['Date']).dt.year <= 2014])
        scalar = MinMaxScaler().fit(trn_df['GHI'].values.reshape(-1,1))

        scalar_tgt = MinMaxScaler().fit(trn_df['Target'].values.reshape(-1,1))
        if self.split == 'train':
            self.df = pd.DataFrame(base_df.loc[pd.to_datetime(base_df['Date']).dt.year <= 2014])
        elif self.split == 'valid':
            self.df = pd.DataFrame(base_df.loc[pd.to_datetime(base_df['Date']).dt.year == 2015])
        else:
            self.df = pd.DataFrame(base_df.loc[pd.to_datetime(base_df['Date']).dt.year > 2015])
            
        self.df['norm_GHI'] = np.squeeze(scalar.transform(self.df['GHI'].values.reshape(-1,1)))
        self.df['norm_Target'] = np.squeeze(scalar_tgt.transform(self.df['Target'].values.reshape(-1,1)))  
        self.x,self.y = self.sliding_windows(self.df,self.seq_length)
        

    def __len__(self):
        return len(self.x)

    def __getitem__(self,ind):
        return self.x[ind].reshape(-1,1),self.y[ind]

    def sliding_windows(self,data, seq_length):
        x = []
        y = []
        dates = sorted(set(data['Date']))
        for date in tqdm(dates):
            #print(date)
            dt_data = data[data['Date'].isin([date])]
            for i in range(len(dt_data)-seq_length-1):
                _x = dt_data.iloc[i:(i+seq_length)]['norm_GHI']
                _y = dt_data.iloc[i+seq_length-1]['norm_Target']
                x.append(_x)
                y.append(_y)
            

        return np.array(x),np.array(y)


class LstmModel(pl.LightningModule):
    def __init__(self,lr,num_classes, input_size, hidden_size, num_layers,seq_length):
        super().__init__()
        self.lstm = LSTM(num_classes, input_size, hidden_size, num_layers,seq_length)
        self.lr = lr
        

        

    def forward(self,x):
        #keep this for inference
        outputs = self.lstm(x.float())
        return outputs

    
    def training_step(self,batch,batch_idx):
        #for training
        #for training
        x,y = batch
        y_hat = torch.squeeze(self.forward(x.float()))


        loss = F.mse_loss(y_hat,y.float())
        self.log("train_loss",loss,on_step=True)
        return loss

    
    def validation_step(self,batch,batch_idx):
        #for training
        x,y = batch
        y_hat = torch.squeeze(self.forward(x.float()))


        loss = F.mse_loss(y_hat,y.float())
        self.log("val_loss",loss,on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        lr = self.lr
        #lr=0.00001, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False
        optimizer = torch.optim.Adam(self.parameters(), lr=lr,betas=(0.9,0.999),amsgrad=False)
        return optimizer


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
               seq_length=cfg.seq_length)
        vld_fdata = Dset(data_dir=cfg.data_dir,split='valid',image_dir= cfg.image_dir,seq_length=cfg.seq_length)

        val_loader = DataLoader(vld_fdata,\
            batch_size=cfg.batch_size,shuffle=False,num_workers=4,pin_memory=True)
        train_loader = DataLoader(trn_fdata,\
            batch_size=cfg.batch_size,shuffle=True,num_workers=4,pin_memory=True)
            
        hparams = cfg    
        net = LstmModel(hparams.lr,hparams.num_classes, hparams.input_size, hparams.hidden_size, hparams.num_layers,hparams.seq_length)
        trainer = pl.Trainer(
            logger=logger_list,callbacks=cbs, gpus=[0],deterministic=True, **cfg.trainer
        )
        trainer.fit(net,train_dataloaders=train_loader,val_dataloaders=val_loader)
        return trainer
        #trainer.tune(net,train_loader)
            
    else:
        pass


if __name__ == '__main__':
    from argparse import Namespace
    cfg = Namespace(
            version = 'lstm',
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
            lr = 1.5e-3,
            batch_size=64,
            input_size = 1,
            hidden_size = 2,
            num_layers = 1,
            num_classes = 1,
            seq_length = 10)
    run_model(cfg)
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
df = pd.read_csv(f'{data_dir}/tgtimgs.csv')

class JDset(Dataset):
    def __init__(self,split= 'train',data_dir='/common/home/vk405/Projects/EnergyLab/Solar-forcast/Data/',
    image_dir='/common/users/vk405/EnergyLab/Data/ProcData',seq_length=10):
        self.split = split

        #hardcoded dir locs 
        self.data_dir = Path(data_dir)
        self.image_dir = Path(image_dir) 
        self.seq_length = seq_length
        
        if 'rawdata.csv' not in os.listdir(data_dir):
            rawdata = self.extract_data()
            rawdata.to_csv(os.path.join(data_dir,'rawdata.csv'),index=False)
        else:
            rawdata = pd.read_csv(data_dir/'rawdata.csv')


        if self.split == 'train':
            self.df = pd.DataFrame(rawdata.loc[pd.to_datetime(rawdata['Date']).dt.year <= 2014])
            
        elif self.split == 'valid':
            self.df = pd.DataFrame(rawdata.loc[pd.to_datetime(rawdata['Date']).dt.year == 2015])
            
        else:
            self.df = pd.DataFrame(rawdata.loc[pd.to_datetime(rawdata['Date']).dt.year > 2015])
 

    def __len__(self):
        return len(self.df)

    def __getitem__(self,ind):
        try:
            imgs = sorted(eval(self.df.iloc[ind]['Imgs']),key = lambda x: int(x.split('/')[-1].split('.')[0]))
            target = self.df.iloc[ind]['10ma_tgt']
            #image_dir = Path('/common/users/vk405/EnergyLab/Data/ProcData') 
            proc_imgs = [self.image_dir/(x.split('/')[-1].split('.')[0]+'.joblib') for x in imgs]
            proc_arrays = [joblib.load(ele) for ele in proc_imgs]
            csis = np.array(self.df.iloc[ind]['CSI'])
            ghis = np.array(self.df.iloc[ind]['GHI'])

        except:
            print(ind)
            
        return (np.concatenate(proc_arrays,-1).reshape(-1,256,256))/255.0,np.stack([csis,ghis],axis=0),target


    def extract_data(self):
        img_df = pd.read_csv(f'{self.data_dir}/tgtimgs.csv')
        base_df = pd.read_csv(os.path.join(self.data_dir, 'Full_SRRL_measurement_timeseries.csv'))
        #generate raw values with lookback
        import datetime as dt
        lkback = self.seq_length
        base_df['DateTime'] = pd.to_datetime(base_df['DateTime'])
        ghis = []
        csis = []
        ys = []
        indices = []
        missing = []
        dates = []
        images = []
        print(f"len before:{len(img_df)}")
        for i in tqdm(range(len(img_df))):
            tgt = img_df.iloc[i]['10ma_tgt']
            date_time = img_df.iloc[i]['DateTime']
            date = img_df.iloc[i]['Date']
            end = dt.datetime.fromisoformat(date_time)
            st = end - dt.timedelta(minutes=lkback)
            chnk = base_df[(base_df['DateTime']>st) &  (base_df['DateTime']<=end)]
            img_loc = img_df.iloc[i]['Imgs']
            if len(chnk) == 10:
                ghi = chnk['GHI'].values
                csi = chnk['10ma_tgt'].values
                ghis.append(ghi)
                csis.append(csi)
                ys.append(tgt)
                indices.append(date_time)
                dates.append(date)
                images.append(img_loc)
            else:
                missing.append(date_time)
        
        
        #normalize everything between 0-1
        clip_ghis = np.clip(np.stack(ghis,axis=0),0.0,1250.0)/1250.0
        clip_csis = np.clip(np.stack(csis,axis=0),0.0,3.8)/3.80
        ys = np.clip(np.array(ys),0.0,3.8)/3.8
        print(f"after len:{len(ys)}")
        return pd.DataFrame({'DateTime':indices,'Date':dates,'Imgs':images,'10ma_tgt':ys,'GHI':clip_ghis.tolist(),'CSI':clip_csis.tolist()})

        

if __name__== "__main__":
    trndset = Dset()

    
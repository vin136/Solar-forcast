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


def extract(all_data,ext_for,lkbk=10):
    ext_vals = []
    for i in tqdm(range(len(ext_for))):
        dt = ' '.join([ext_for.iloc[i]['Date'],ext_for.iloc[i]['MST']])
        ind = all_data[all_data['DateTime'] == dt].index.tolist()[0]
        ext_vals.append(all_data.iloc[ind-lkbk+1:ind+1]['GHI'].values)
    return np.concatenate(ext_vals)




if __name__== '__main__':
    trndset = Dset()
    df_svd = pd.read_csv(os.path.join(data_dir, 'SRRL_measurement_timeseries.csv'))
    trn_msi = extract(df_svd,trndset.df)
    joblib.dump(trn_msi,data_dir/'trn_msi.joblib')

    vlddset = Dset(split='valid')
    vld_msi = extract(df_svd,vlddset.df)
    joblib.dump(vld_msi,data_dir/'vld_msi.joblib')

    tstdset = Dset(split='test')
    tst_msi = extract(df_svd,tstdset.df)
    joblib.dump(tst_msi,data_dir/'tst_msi.joblib')

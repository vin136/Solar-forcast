#inference code
from base import *
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
from pathlib import Path
data_dir =Path('/common/home/vk405/Projects/EnergyLab/Solar-forcast/Data/')
image_dir = Path('/common/users/vk405/EnergyLab/Data')
proc_data_dir = Path('/common/users/vk405/EnergyLab/Data/ProcData')
df = pd.read_csv(f'{data_dir}/tgtimgs.csv')

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


def infer(model,loader,cfg=''):
    preds_lis = []
    truelabels = []
    with torch.no_grad():
        if cfg and cfg.model_type == 'joint':
            for x_img,x_lkbk,y in tqdm(test_loader):
                preds = model(x_img.float(),x_lkbk.float())
                preds_lis.append(preds.squeeze().cpu().numpy())
                truelabels.append(y.cpu().numpy())
        else:
            for x,y in tqdm(test_loader):
                preds = model(x.float())
                preds_lis.append(preds.squeeze().cpu().numpy())
                truelabels.append(y.cpu().numpy())
    return np.concatenate(preds_lis,axis=0),np.concatenate(truelabels,axis=0)
    


if __name__ == '__main__':
    #just change config and the weight_loc
    from argparse import Namespace
    data_dir =Path('/common/home/vk405/Projects/EnergyLab/Solar-forcast/Data/')
    cfg = Namespace(
    version = 'joint_csi_tgt_ghi',
    model_type = 'joint',
    artifacts_loc = "/common/home/vk405/Projects/EnergyLab/Solar-forcast/artifacts/",
    data_dir = "/common/home/vk405/Projects/EnergyLab/Solar-forcast/Data/",
    image_dir = "/common/users/vk405/EnergyLab/Data/ProcData/",
    mode = 'train',
    loggers = ["csv"],
    seed = 0,
    cbs = ["checkpoint"],
    trainer = {'log_every_n_steps': 50,
    'max_epochs': 50},
    checkpoint = {"save_top_k": 5,
    "monitor": "val_loss","mode":"min"},
    dropoutp=0.2,
    framecnt=2,
    lr = 1.5e-5,
    batch_size=64,
    input_size = 2,
    hidden_size = 4,
    num_layers = 1,
    num_classes = 1,
    seq_length = 10,
    tgt = 'ghi'
)   
    if 'model_type' in cfg and cfg.model_type == 'joint':
        test_dset = JDset(data_dir=cfg.data_dir,split='test',image_dir= cfg.image_dir,seq_length=cfg.seq_length)
        net = JointModel(cfg)
    else:
        test_dset = Dset(data_dir=cfg.data_dir,split='test',image_dir= cfg.image_dir\
        )
        net = VggModel(cfg)

    
    weight_loc = f'/common/home/vk405/Projects/EnergyLab/Solar-forcast/artifacts/ckpts/{cfg.version}/epoch=48-val_loss=0.09.ckpt'
    
    trnd_net = net.load_from_checkpoint(weight_loc)
    test_loader = DataLoader(test_dset,\
            batch_size=64,shuffle=False,num_workers=4,pin_memory=True)
    p,g = infer(trnd_net,test_loader,cfg)
    if 'model_type' in cfg and cfg.model_type == 'joint':
        if 'tgt' not in cfg or ('tgt' in cfg and cfg.tgt != 'ghi'):
            infered_vals = {'Date':test_dset.df['Date'].values,'DateTime':test_dset.df['DateTime'].values,
    'pred':p,'ground':g,'10ma_tgt':test_dset.df['10ma_tgt'].values}
        else:
            infered_vals = {'Date':test_dset.df['Date'].values,'DateTime':test_dset.df['DateTime'].values,
    'pred':p,'ground':g,'10ma_tgt_GHI':test_dset.df['10ma_tgt_GHI'].values}

    else:
        infered_vals = {'Date':test_dset.df['Date'].values,'MST':test_dset.df['MST'].values,
    'pred':p,'ground':g,'Target':test_dset.df['Target'].values}
    pd.DataFrame(infered_vals).to_csv(data_dir/f'{cfg.version}_infered_vals.csv',index=False)

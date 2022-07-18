from base import *
from argparse import Namespace




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
        print('inside train')
        hparams = cfg 
        tgt = cfg.tgt if 'tgt' in cfg else 'csi'
        print(f"training on :{tgt}")
        if 'model_type' in cfg and cfg.model_type == 'joint':
            trn_fdata = JDset(data_dir=cfg.data_dir,split=cfg.mode,image_dir= cfg.image_dir,seq_length=cfg.seq_length,tgt=tgt)
            vld_fdata = JDset(data_dir=cfg.data_dir,split='valid',image_dir= cfg.image_dir,seq_length=cfg.seq_length,tgt=tgt)
            net = JointModel(hparams)
        else:
            trn_fdata = Dset(data_dir=cfg.data_dir,split=cfg.mode,image_dir= cfg.image_dir\
               )
            vld_fdata = Dset(data_dir=cfg.data_dir,split='valid',image_dir= cfg.image_dir)
            net = VggModel(hparams)

        val_loader = DataLoader(vld_fdata,\
            batch_size=cfg.batch_size,shuffle=False,num_workers=4,pin_memory=True)
        train_loader = DataLoader(trn_fdata,\
            batch_size=cfg.batch_size,shuffle=True,num_workers=4,pin_memory=True)
            
         
        print('Dsets loaded ')  
        
        trainer = pl.Trainer(
            logger=logger_list,callbacks=cbs, gpus=[0,1,2,3],deterministic=True, **cfg.trainer
        )
        trainer.fit(net,train_dataloaders=train_loader,val_dataloaders=val_loader)
        return trainer
        #trainer.tune(net,train_loader)
            
    else:
        pass
    

#model type = 'cnn','joint'
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


if __name__ == '__main__':
    from argparse import Namespace
    run_model(cfg)
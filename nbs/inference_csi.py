from base import *
from argparse import Namespace

def infer(model,loader):
    preds_lis = []
    truelabels = []
    with torch.no_grad():
        for x,y in tqdm(test_loader):
            preds = model(x.float())
            preds_lis.append(preds.squeeze().cpu().numpy())
            truelabels.append(y.cpu().numpy())
    return np.concatenate(preds_lis,axis=0),np.concatenate(truelabels,axis=0)


if __name__ == '__main__':
    cfg = Namespace(
    version = 'img_csi',
    artifacts_loc = "/common/home/vk405/Projects/EnergyLab/Solar-forcast/artifacts/",
    data_dir = "/common/home/vk405/Projects/EnergyLab/Solar-forcast/Data/",
    image_dir = "/common/users/vk405/EnergyLab/Data/ProcData/",
    mode = 'train',
    loggers = ["csv"],
    seed = 0,
    cbs = ["checkpoint"],
    trainer = {'log_every_n_steps': 50,
    'max_epochs': 30},
    checkpoint = {"save_top_k": 5,
    "monitor": "val_loss","mode":"min"},
    dropoutp=0.2,
    framecnt=2,
    lr = 1.5e-5,
    batch_size=64)   
    test_dset = Dset(split= 'test')
    #trained model
    net = VggModel(cfg)
    weight_loc = '/common/home/vk405/Projects/EnergyLab/Solar-forcast/artifacts/ckpts/img_csi/epoch=27-val_loss=0.05.ckpt'
    trnd_net = net.load_from_checkpoint(weight_loc)
    test_loader = DataLoader(test_dset,\
            batch_size=64,shuffle=False,num_workers=4,pin_memory=True)
    p,g = infer(trnd_net,test_loader)
    infered_vals = {'Date':test_dset.df['Date'].values,'MST':test_dset.df['MST'].values,
    'DateTime':test_dset.df['DateTime'].values,'pred_tgt':p,'ground_tgt':g,'10ma_tgt':test_dset.df['10ma_tgt'].values}
    pd.DataFrame(infered_vals).to_csv(data_dir/'infered_vals_csi.csv',index=False)


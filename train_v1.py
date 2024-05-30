import h5py
import torch
from torch.utils.data import DataLoader
import configargparse
from utils.lose import get_loss_s_1
from utils.data_loading import MyDataset
from utils.utils import snr
from net.VAE3 import FCEncoder1, FCDecoder1, VAE
import wandb
from pylab import *
# wandb offline
from sklearn.preprocessing import Normalizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(824)  # 固定随机种子（CPU）
if torch.cuda.is_available():  # 固定随机种子（GPU)
    torch.cuda.manual_seed(824)
wandb.init(project="czkl_v1", entity="shupiqiu")

p = configargparse.ArgumentParser()
p.add_argument('--trainfile_location', type=str, default='/home/fiber/stu_code/data/VAESNR_58W.h5', help='')
p.add_argument('--testfile_location', type=str, default='/home/fiber/stu_code/data/test/snr/VAESNR_F50.h5', help='')


p.add_argument('--integrity', type=float, default=1, help='')
p.add_argument('--snr_num', type=int, default=9, help='')
p.add_argument('--snr_min', type=int, default=4, help='')
p.add_argument('--snr_idx', type=int, default=1, help='')

p.add_argument('--epochs_s_1', type=int, default=350, help='epochs in stage 1 training')
p.add_argument('--epochs_pr', type=int, default=1, help='epochs in stage 1、2 print')#190，301

p.add_argument('--batch_size', type=int, default=100, help='')#500,300,100
p.add_argument('--batch_size_test', type=int, default=100, help='')

p.add_argument('--lr1', type=float, default=4e-4 , help='')#-3 太大了，出现nan   4e-5,9e-5,9e-4,3e-4!7e-4太大

p.add_argument('--data_len', type=int, default=300, help='')
p.add_argument('--latent_features', type=int, default=10, help='')

p.add_argument('--v1_per1', type=float, default=10, help='mse')
p.add_argument('--v1_per2', type=float, default=1, help='kld')
p.add_argument('--v1_per3', type=float, default=0, help='bfs')


opt = p.parse_args()
wandb.config.update(opt)
# Read the file and Save the two dataset in two variables
archive_train = h5py.File(opt.trainfile_location, 'r')
train_data_np = archive_train['train_data'][:]
row_data_np = archive_train['row_data'][:]
vb_data_np = archive_train['train_label1'][:]
snr_data_np = archive_train['train_label2'][:]
train_set = MyDataset(opt.integrity,train_data_np, row_data_np, vb_data_np, snr_data_np,transformer=None)
#
archive_test = h5py.File(opt.testfile_location, 'r')
t_test = archive_test['train_data'][:]
r_test = archive_test['row_data'][:]
vb_test = archive_test['train_label1'][:]
snr_test = archive_test['train_label2'][:]
test_set = MyDataset(opt.integrity,t_test, r_test, vb_test, snr_test,transformer=None)



loader_args = dict(batch_size=opt.batch_size, num_workers=16, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True,drop_last=True, **loader_args)
loader_args = dict(batch_size=opt.batch_size_test, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_args)



Encoder_1 = FCEncoder1(input_dim=opt.data_len, latent_size=opt.latent_features).to(device=device)
Decoder_1 = FCDecoder1(input_dim=opt.data_len, latent_size=opt.latent_features).to(device=device)
VAE_1 = VAE(Encoder_1, Decoder_1).to(device=device)
# Encoder_1.load_state_dict(torch.load(opt.v1EN))
# Decoder_1.load_state_dict(torch.load(opt.v1DE))
# VAE_1.load_state_dict(torch.load(opt.v1model))

optimizer1 = torch.optim.Adam([
    {'params': Encoder_1.parameters(), 'lr': opt.lr1},
    {'params': Decoder_1.parameters(), 'lr': opt.lr1}
    # {'params': VAE_1.parameters(), 'lr': opt.lr1}
])

v1index = 0
min_mse=10000
for epoch in range(opt.epochs_s_1+1):

    Encoder_1.train()
    Decoder_1.train()
    VAE_1.train()
    Encoder_1.requires_grad_(True)
    Decoder_1.requires_grad_(True)
    # VAE_1.requires_grad_(True)

    mu = None
    logvar = None
    batchidx= 0

    for batch in train_loader:
        batchidx=batchidx+1
        V1_data = batch['t_data'].to(device=device, dtype=torch.float32)
        V1_row_data = batch['r_data'].to(device=device, dtype=torch.float32)
        V1_data = torch.squeeze(V1_data)
        V1_row_data = torch.squeeze(V1_row_data)
        # V1_data_c=V1_data+1
        x_hat, mu, logvar, z = VAE_1(V1_data)
        # x_hat=x_hat_c-1

        # V1_data = torch.squeeze(V1_data)
        # x_hat = torch.squeeze(x_hat)
        # mu = torch.squeeze(mu)
        # logvar = torch.squeeze(logvar)
        # z = torch.squeeze(z)
        # V1_row_data = torch.squeeze(V1_row_data)
        MSE_L1, KLD, loss_s_1, bfs_RMSE,r2_score = get_loss_s_1(x_hat,V1_row_data,mu,logvar,opt.v1_per1,opt.v1_per2,opt.v1_per3,device)
        optimizer1.zero_grad()
        loss_s_1.backward()
        optimizer1.step()
        with torch.no_grad():
            wandb.log({"loss": loss_s_1})
            wandb.log({"bfs": bfs_RMSE})
            wandb.log({"MSE_L": MSE_L1})
            wandb.log({"KLD": KLD})
            wandb.log({"r2_score": r2_score})

    with torch.no_grad():
        Encoder_1.eval()
        Decoder_1.eval()
        VAE_1.eval()
        if epoch != 0 and epoch % opt.epochs_pr == 0:
            print("epo{}, loss:{}, MSE_L:{}, KLD:{}, bfs:{}".format(epoch,loss_s_1, MSE_L1, KLD, bfs_RMSE))
            flag,test_value=snr(VAE_1, test_loader, opt.integrity, opt.snr_num, opt.snr_min, opt.snr_idx,min_mse, device)
            # flag,test_value=snr(VAE_1, test_loader1,test_loader2,test_loader3 opt.integrity, opt.snr_num, opt.snr_min, opt.snr_idx,min_mse, device)
            if flag==1:
                min_mse=test_value
                torch.save(VAE_1.state_dict(), '/home/fiber/stu_code/czklvae/ckpt/1/t/'+'V1_'+str(epoch)+'_'+wandb.run.name[-3:]+'.ckpt')
                torch.save(Encoder_1.state_dict(), '/home/fiber/stu_code/czklvae/ckpt/1/t/'+'V1EN_'+str(epoch)+'_'+wandb.run.name[-3:]+'.ckpt')
                torch.save(Decoder_1.state_dict(), '/home/fiber/stu_code/czklvae/ckpt/1/t/'+'V1DE_'+str(epoch)+'_'+wandb.run.name[-3:]+'.ckpt')


            elif flag==2:
                torch.save(VAE_1.state_dict(), '/home/fiber/stu_code/czklvae/ckpt/1/t/'+'V1_'+str(epoch)+'_'+wandb.run.name[-3:]+'.ckpt')
                torch.save(Encoder_1.state_dict(), '/home/fiber/stu_code/czklvae/ckpt/1/t/'+'V1EN_'+str(epoch)+'_'+wandb.run.name[-3:]+'.ckpt')
                torch.save(Decoder_1.state_dict(), '/home/fiber/stu_code/czklvae/ckpt/1/t/'+'V1DE_'+str(epoch)+'_'+wandb.run.name[-3:]+'.ckpt')



















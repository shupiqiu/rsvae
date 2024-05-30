import h5py
import torch
from torch.utils.data import DataLoader
import configargparse
from utils.lose import get_loss_2_eval
from utils.data_loading import MyDataset
from utils.utils import snr
from net.VAE3 import FCEncoder1, FCDecoder1, VAE
import wandb


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(824)  # 固定随机种子（CPU）
if torch.cuda.is_available():  # 固定随机种子（GPU)
    torch.cuda.manual_seed(824)

wandb.init(project="czkl_v2", entity="shupiqiu")

p = configargparse.ArgumentParser()
p.add_argument('--trainfile_location', type=str, default='/home/fiber/stu_code/data/VAESNR_58W.h5', help='')
p.add_argument('--testfile_location', type=str, default='/home/fiber/stu_code/data/test/snr/VAESNR_F50.h5', help='')
p.add_argument('--integrity', type=float, default=0.5, help='')
p.add_argument('--snr_num', type=int, default=9, help='')
p.add_argument('--snr_min', type=int, default=4, help='')
p.add_argument('--snr_idx', type=int, default=1, help='')

p.add_argument('--v1model', type=str, default='/home/fiber/stu_code/czklvae/ckpt/1/t/V1_6_569.ckpt', help='')
p.add_argument('--v1EN', type=str, default='/home/fiber/stu_code/czklvae/ckpt/1/t/V1EN_6_569.ckpt', help='')
p.add_argument('--v1DE', type=str, default='/home/fiber/stu_code/czklvae/ckpt/1/t/V1DE_6_569.ckpt', help='')


p.add_argument('--epochs_s_2', type=int, default=500, help='epochs in stage 2 training')#190，301
p.add_argument('--epochs_pr', type=int, default=1, help='epochs in stage 1、2 print')#190，301

p.add_argument('--batch_size', type=int, default=100, help='')#500,300,100
p.add_argument('--batch_size_test', type=int, default=100, help='')

p.add_argument('--lr1', type=float, default=2e-5, help='')

p.add_argument('--data_len', type=int, default=300, help='')
p.add_argument('--latent_features', type=int, default=10, help='')

p.add_argument('--v2_per1', type=float, default=1, help='l1 mse')
p.add_argument('--v2_per2', type=float, default=0, help='kld_standard_DEL')
p.add_argument('--v2_per3', type=float, default=1, help='kld_czkl')
p.add_argument('--ev_per1', type=float, default=1, help='mse_L')
p.add_argument('--ev_per2', type=float, default=0, help='RMSE')


opt = p.parse_args()
wandb.config.update(opt)

archive_train = h5py.File(opt.trainfile_location, 'r')
train_data_np = archive_train['train_data'][:]
row_data_np = archive_train['row_data'][:]
vb_data_np = archive_train['train_label1'][:]
snr_data_np = archive_train['train_label2'][:]
#
archive_test = h5py.File(opt.testfile_location, 'r')
train_data_np_test = archive_test['train_data'][:]
row_data_np_test = archive_test['row_data'][:]
vb_data_np_test = archive_test['train_label1'][:]
snr_data_np_test = archive_test['train_label2'][:]
train_set = MyDataset(opt.integrity,train_data_np, row_data_np, vb_data_np, snr_data_np,transformer=None)
test_set = MyDataset(opt.integrity,train_data_np_test, row_data_np_test, vb_data_np_test, snr_data_np_test,transformer=None)
loader_args = dict(batch_size=opt.batch_size, num_workers=16, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True,drop_last=True, **loader_args)
loader_args = dict(batch_size=opt.batch_size_test, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_set, shuffle=True, drop_last=False, **loader_args)


Encoder_1 = FCEncoder1(input_dim=opt.data_len, latent_size=opt.latent_features).to(device=device)
Decoder_1 = FCDecoder1(input_dim=opt.data_len, latent_size=opt.latent_features).to(device=device)
Encoder_k05 = FCEncoder1(input_dim=int(opt.integrity*opt.data_len), latent_size=opt.latent_features).to(device=device)
Decoder_k05 = FCDecoder1(input_dim=int(opt.integrity*opt.data_len), latent_size=opt.latent_features).to(device=device)

VAE_1 = VAE(Encoder_1, Decoder_1).to(device=device)
VAE_k05= VAE(Encoder_k05, Decoder_k05).to(device=device)
eval_VAE_k05 = VAE(Encoder_k05, Decoder_1).to(device=device)
Encoder_1.load_state_dict(torch.load(opt.v1EN))
Decoder_1.load_state_dict(torch.load(opt.v1DE))
VAE_1.load_state_dict(torch.load(opt.v1model))

optimizerk = torch.optim.Adam([
    {'params': Encoder_1.parameters(), 'lr': opt.lr1},
    {'params': Decoder_1.parameters(), 'lr': opt.lr1},
    {'params': Encoder_k05.parameters(), 'lr': opt.lr1},
    {'params': Decoder_k05.parameters(), 'lr': opt.lr1}
])


epoch_step = 0
v1index = 0
min_mse=1000



for epoch1 in range(opt.epochs_s_2+1):
    # Encoder_1.eval()
    # Decoder_1.eval()
    # Encoder_1.requires_grad_(False)
    # Decoder_1.requires_grad_(False)
    Encoder_1.train()
    Decoder_1.train()
    Encoder_1.requires_grad_(True)
    Decoder_1.requires_grad_(True)
    Encoder_k05.train()
    Decoder_k05.train()
    Encoder_k05.requires_grad_(True)
    Decoder_k05.requires_grad_(True)
    eval_VAE_k05.train()
    eval_VAE_k05.requires_grad_(True)
    VAE_k05.train()
    VAE_k05.requires_grad_(True)
    validx = 0
    i=0
    batchidx= 0
    mu1 = None
    lv1 = None
    mu0 = None
    lv0 = None

    for batch_1 in train_loader:
        batchidx = batchidx+1
        V2_data = batch_1['t_data'].to(device=device, dtype=torch.float32)
        V2_row_data = batch_1['r_data'].to(device=device, dtype=torch.float32)
        V2_masked_data = batch_1['t_idata'].to(device=device, dtype=torch.float32)
        V2_masked_row_data = batch_1['r_idata'].to(device=device, dtype=torch.float32)

        V2_data = torch.squeeze(V2_data)
        V2_row_data = torch.squeeze(V2_row_data)
        V2_masked_data = torch.squeeze(V2_masked_data)
        V2_masked_row_data = torch.squeeze(V2_masked_row_data)

        # V2_data_c=V2_data+1
        # V2_masked_data_c=V2_masked_data+1
        _, mu1, lv1,_ = VAE_1(V2_data)
        ixhat, mu0, lv0, _ = VAE_k05(V2_masked_data)
        xhat,_,_,_= eval_VAE_k05(V2_masked_data)
        # ixhat=ixhat_c-1
        # xhat=xhat_c-1


        v2MSEL,KLD1,KLD2,delKLD,czKLD,evMSEL,loss,evRMSE=\
            get_loss_2_eval(ixhat,V2_masked_row_data,xhat,V2_row_data,mu0,lv0,mu1,lv1,opt.v2_per1,opt.v2_per2,opt.v2_per3,opt.ev_per1,opt.ev_per2, device)

        optimizerk.zero_grad()
        loss.backward()
        optimizerk.step()
        with torch.no_grad():
            wandb.log({"loss": loss})
            wandb.log({"MSE_L": v2MSEL})
            wandb.log({"KLDCZK": czKLD})
            wandb.log({"ev_MSE": evMSEL})
            wandb.log({"DELKLD": delKLD})

    with torch.no_grad():
        Encoder_1.eval()
        Decoder_1.eval()
        Encoder_k05.eval()
        Decoder_k05.eval()
        eval_VAE_k05.eval()
        VAE_k05.eval()
        if epoch1 % opt.epochs_pr == 0 and epoch1 != 0:
            print("epo:{}, loss:{}, MSE_L:{}, KLD:{},CZK:{},V1KLD:{},ev_MSEL:{},evRMSE:{}".format(epoch1, loss, v2MSEL, KLD2,czKLD, KLD1,evMSEL,evRMSE))
            flag, test_value = snr(eval_VAE_k05, test_loader, opt.integrity, opt.snr_num, opt.snr_min, opt.snr_idx,min_mse,device)
            if flag==1:
                min_mse=test_value
                torch.save(VAE_k05.state_dict(),  '/home/fiber/stu_code/czklvae/ckpt/K05/'+'V2_'+str(epoch1)+'_'+str(opt.integrity)+'_' + wandb.run.name[-3:] + '.ckpt')
                torch.save(eval_VAE_k05.state_dict(),  '/home/fiber/stu_code/czklvae/ckpt/K05/'+'ev_' + str(epoch1)+'_'+str(opt.integrity)+'_' +wandb.run.name[-3:] + '.ckpt')
            elif flag==2:
                torch.save(VAE_k05.state_dict(),  '/home/fiber/stu_code/czklvae/ckpt/K05/'+'V2_'+str(epoch1)+'_'+str(opt.integrity)+'_'+wandb.run.name[-3:] + '.ckpt')
                torch.save(eval_VAE_k05.state_dict(),  '/home/fiber/stu_code/czklvae/ckpt/K05/'+'ev_' + str(epoch1)+'_'+str(opt.integrity)+'_'+wandb.run.name[-3:] + '.ckpt')
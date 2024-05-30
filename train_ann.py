import h5py
import torch
import utils.PartialSimulationLibz as psl
from torch.utils.data import DataLoader, random_split
import configargparse
from utils.data_loading import MyDataset
from net.VAE2 import FCEncoder1, FCDecoder1, VAE
from utils.nets import ANN, RANet,DeepResNet
import wandb
from utils.utils import bfs_integrity,out_in,snr,vae_out_in,label_snr
from sklearn.preprocessing import Normalizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(824)  # 固定随机种子（CPU）
if torch.cuda.is_available():  # 固定随机种子（GPU)
    torch.cuda.manual_seed(824)
wandb.init(project="czkl_ann", entity="shupiqiu")
# Command line argument processing
p = configargparse.ArgumentParser()

p.add_argument('--trainfile_location', type=str, default='/home/fiber/stu_code/data/WITHLABOR.h5', help='')
p.add_argument('--testfile_location', type=str, default='/home/fiber/stu_code/data/test/snr/VAESNR_F50.h5', help='')
p.add_argument('--integrity', type=float, default=0.5, help='')

p.add_argument('--snr_num', type=int, default=9, help='')
p.add_argument('--snr_min', type=int, default=4, help='')
p.add_argument('--snr_idx', type=int, default=1, help='')
p.add_argument('--evmodel', type=str, default='/home/fiber/stu_code/czklvae/ckpt/1/change/496/V1_3_496.ckpt', help='')

p.add_argument('--Net_name', type=str,  default='RANet', help='ANN,RANet,ResNet')#190，301
p.add_argument('--label_name', type=str,  default='BFS', help='BFS,FWHM')#190，301
p.add_argument('--epochs_Net', type=int, default=2000, help='epochs in stage 2 training')#190，301
p.add_argument('--epochs_Net_val', type=int, default=1, help='epochs in stage 2 training')#190，301

p.add_argument('--batch_size_Net', type=int, default=100, help='')#500,300,100
p.add_argument('--batch_size_test', type=int, default=100, help='')

p.add_argument('--lrNet', type=float, default=5e-4, help='')#-3 太大了，出现nan   4e-5,9e-5,9e-4,3e-4!7e-4太大


p.add_argument('--data_len', type=float, default=300, help='')
p.add_argument('--latent_features', type=float, default=10, help='')


# parse arguments
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
loader_args = dict(batch_size=opt.batch_size_Net, num_workers=16, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
loader_args = dict(batch_size=opt.batch_size_test, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)

Decoder_1 = FCDecoder1(input_dim=opt.data_len, latent_size=opt.latent_features).to(device=device)
# Encoder_k05 = FCEncoder1(input_dim=int(opt.data_len*opt.integrity),latent_size=opt.latent_features).to(device=device)
Encoder_k05 = FCEncoder1(input_dim=int(opt.data_len),latent_size=opt.latent_features).to(device=device)
eval_VAE = VAE(Encoder_k05, Decoder_1).to(device=device)
eval_VAE.load_state_dict(torch.load(opt.evmodel))

if opt.Net_name == "ResNet":
    net0 = DeepResNet().to(device=device)
    net1 = DeepResNet().to(device=device)
elif opt.Net_name == "ANN":
    net0 = ANN(input_dim=opt.data_len).to(device=device)
    net1 = ANN(input_dim=opt.data_len).to(device=device)
elif opt.Net_name == "RANet":
    net0 = RANet(block_channels=64,input_dim=opt.data_len).to(device=device)
    net1 = RANet(block_channels=64,input_dim=opt.data_len).to(device=device)

optimizer = torch.optim.Adam([
    {'params': net0.parameters(), 'lr': opt.lrNet}
])
min_mse=300
for epoch in range(opt.epochs_Net+1):
    # eval_VAE.eval()
    # eval_VAE.requires_grad_(False)

    net0.train()
    net0.requires_grad_(True)
    for Net_batch in train_loader:
        net_masked_t_data = Net_batch['t_idata'].to(device=device, dtype=torch.float32)  # , dtype=torch.float32
        net_masked_t_data = net_masked_t_data.squeeze(dim=1)

        # -------------eval_VAE--------------------#
        # net_masked_t_data_c=net_masked_t_data+1
        # vae_pre_data, _, _, _ = eval_VAE(net_masked_t_data)
        # vae_pre_data=vae_pre_data_c-1
        # -------------VAE+ANN--------------------#
        # if opt.Net_name == "ResNet":
        #     PRE_net= net0(vae_pre_data.unsqueeze(dim=1)).squeeze()
        # elif opt.Net_name == "ANN":
        #     PRE_net = net0(vae_pre_data).squeeze()
        # elif opt.Net_name == "RANet":
        #     PRE_net = net0(vae_pre_data.unsqueeze(dim=1)).squeeze()
        if opt.Net_name == "ResNet":
            PRE_net = net0(net_masked_t_data.unsqueeze(dim=1)).squeeze()
        elif opt.Net_name == "ANN":
            PRE_net = net0(net_masked_t_data).squeeze()
        elif opt.Net_name == "RANet":
            PRE_net = net0(net_masked_t_data.unsqueeze(dim=1)).squeeze()
        net_BFS = Net_batch['label1'].to(device=device, dtype=torch.float32) - 10700
        net_BFS = torch.squeeze(net_BFS)
        RMSE_VAEANN = torch.sqrt(torch.sum(torch.pow(torch.abs(PRE_net - net_BFS).to(dtype=torch.float32), 2)))


        #
        # if opt.label_name == "BFS":
        #     net_BFS = Net_batch['label1'].to(device=device, dtype=torch.float32) - 10700
        #     net_BFS = torch.squeeze(net_BFS)
        #     RMSE_VAEANN = torch.sqrt(torch.sum(torch.pow(torch.abs(PRE_net - net_BFS).to(dtype=torch.float32), 2)))
        #
        # elif opt.label_name == "FWHM":
        #     net_FWHM = Net_batch['label3'].to(device=device, dtype=torch.float32)
        #     net_FWHM = torch.squeeze(net_FWHM)
        #     MSE_VAEANN = torch.sum(torch.pow(torch.abs(PRE_net - net_FWHM).to(dtype=torch.float32), 2))
        #     RMSE_VAEANN = torch.sqrt(MSE_VAEANN)

        optimizer.zero_grad()
        RMSE_VAEANN.backward()
        optimizer.step()
        with torch.no_grad():
            wandb.log({opt.label_name+"vaeann_rmse": RMSE_VAEANN})
    with torch.no_grad():
        torch.save(net0.state_dict(), '/home/fiber/stu_code/czklvae/ckpt/BFS/' +opt.Net_name + wandb.run.name[-3:] + '.ckpt')
        net1.load_state_dict(torch.load('/home/fiber/stu_code/czklvae/ckpt/BFS/' +opt.Net_name + wandb.run.name[-3:] + '.ckpt'))
        net1.eval()
        # eval_VAE.eval()
        if epoch % opt.epochs_Net_val == 0:
            print("epo{},rmse:{}".format(epoch, RMSE_VAEANN))
            flag, test_value = label_snr(eval_VAE, net1, psl.LorenzFit, psl.PVFit, opt.Net_name, opt.label_name,test_loader,
                                         opt.integrity, opt.snr_num, opt.snr_min, opt.snr_idx, min_mse,device)
            if flag==1:
                min_mse=test_value
                torch.save(net1.state_dict(), '/home/fiber/stu_code/czklvae/ckpt/BFS/' + str(epoch) +'_'+ opt.label_name +'_'+ opt.Net_name + wandb.run.name[-3:] + '.ckpt')
            elif flag==2:
                torch.save(net1.state_dict(), '/home/fiber/stu_code/czklvae/ckpt/BFS/' + str(epoch) +'_'+ opt.label_name +'_'+ opt.Net_name + wandb.run.name[-3:] + '.ckpt')














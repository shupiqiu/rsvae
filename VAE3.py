import torch
from torch import nn
from torch.nn import Module


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
# 3x1 convolution
def conv3x1(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)
def conv15x1(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=15,
                     stride=stride, padding=7, bias=True)
# 3x1 convolution
def conv16x1(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=16,
                     stride=stride, padding=7, bias=True)
def convT3x1(in_channels, out_channels, stride=1):
    return nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)
def convT1x1(in_channels, out_channels, stride=1):
    return nn.ConvTranspose1d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)
def convT15x1(in_channels, out_channels, stride=1):
    return nn.ConvTranspose1d(in_channels, out_channels, kernel_size=15,
                     stride=stride, padding=7, bias=True)
# 3x1 convolution
def convT16x1(in_channels, out_channels, stride=1):
    return nn.ConvTranspose1d(in_channels, out_channels, kernel_size=16,
                     stride=stride, padding=1, bias=True)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(nn.Conv1d(in_planes, in_planes // 16, 1, bias=False),
                                nn.SELU(),
                                nn.Conv1d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CBAM, self).__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.SELU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        output = residual + out

        return output
class FCM_ResBlock(nn.Module):
    def __init__(self, block_channels):
        super(FCM_ResBlock, self).__init__()
        self.block_channels = block_channels

        self.conv1 = conv1x1(self.block_channels, int(self.block_channels/4))
        self.bn1 = nn.BatchNorm1d(int(self.block_channels/4))
        self.conv2 = conv3x1(int(self.block_channels/4), int(self.block_channels/4))
        self.bn2 = nn.BatchNorm1d(int(self.block_channels/4))
        self.conv3 = conv1x1(int(self.block_channels/4), int(self.block_channels/4))
        self.bn3 = nn.BatchNorm1d(int(self.block_channels/4))

        self.conv_skip = conv1x1(self.block_channels, int(self.block_channels/4))
        self.bn_skip = nn.BatchNorm1d(int(self.block_channels/4))

        self.conv4 = conv3x1(int(self.block_channels/4), int(self.block_channels/4))
        self.bn4 = nn.BatchNorm1d(int(self.block_channels/4))
        self.relu = torch.nn.SELU()


    def forward(self, x):
        residual = self.conv_skip(x)
        residual = self.bn_skip(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        output = residual + out

        output = self.conv4(output)
        output = self.bn4(output)

        return output
class FEM_RABlock(nn.Module):
    def __init__(self, block_channels):
        super(FEM_RABlock, self).__init__()
        self.block_channels = block_channels

        self.conv1 = conv1x1(self.block_channels, self.block_channels)
        self.bn1 = nn.BatchNorm1d(self.block_channels)
        self.conv2 = conv3x1(self.block_channels, self.block_channels)
        self.bn2 = nn.BatchNorm1d(self.block_channels)
        self.conv3 = conv1x1(self.block_channels, self.block_channels)
        self.bn3 = nn.BatchNorm1d(self.block_channels)
        self.cbam = CBAM(self.block_channels, self.block_channels)

        self.conv4 = conv3x1(self.block_channels, 2*self.block_channels, 2)
        self.bn4 = nn.BatchNorm1d(2*self.block_channels)
        self.relu = torch.nn.SELU()


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.cbam(out)

        output = residual + out

        output = self.conv4(output)
        output = self.bn4(output)


        return output
class CBAMT(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CBAMT, self).__init__()
        self.conv1 = convT3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.SELU(inplace=True)
        self.conv2 = convT3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        output = residual + out

        return output
class FCM(nn.Module):
    def __init__(self, block_channels):
        super(FCM, self).__init__()
        self.block_channels = block_channels

        self.conv1 = convT1x1(self.block_channels, int(self.block_channels*4))
        self.bn1 = nn.BatchNorm1d(int(self.block_channels*4))
        self.conv2 = convT3x1(int(self.block_channels*4), int(self.block_channels*4))
        self.bn2 = nn.BatchNorm1d(int(self.block_channels*4))
        self.conv3 = convT1x1(int(self.block_channels*4), int(self.block_channels*4))
        self.bn3 = nn.BatchNorm1d(int(self.block_channels*4))

        self.conv_skip = convT1x1(self.block_channels, int(self.block_channels*4))
        self.bn_skip = nn.BatchNorm1d(int(self.block_channels*4))

        self.conv4 = convT3x1(int(self.block_channels*4), int(self.block_channels*4))
        self.bn4 = nn.BatchNorm1d(int(self.block_channels*4))
        self.relu = torch.nn.SELU()


    def forward(self, x):
        residual = self.conv_skip(x)
        residual = self.bn_skip(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        output = residual + out

        output = self.conv4(output)
        output = self.bn4(output)

        return output
class FEM(nn.Module):
    def __init__(self, block_channels):
        super(FEM, self).__init__()
        self.block_channels = block_channels

        self.conv1 = convT1x1(self.block_channels, self.block_channels)
        self.bn1 = nn.BatchNorm1d(self.block_channels)
        self.conv2 = convT3x1(self.block_channels, self.block_channels)
        self.bn2 = nn.BatchNorm1d(self.block_channels)
        self.conv3 = convT1x1(self.block_channels, self.block_channels)
        self.bn3 = nn.BatchNorm1d(self.block_channels)
        self.cbam = CBAMT(self.block_channels, self.block_channels)
        self.conv4 = convT3x1(self.block_channels, int(self.block_channels/2), 2)
        self.bn4 = nn.BatchNorm1d(int(self.block_channels/2))
        self.relu = torch.nn.SELU()


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.cbam(out)

        output = residual + out

        output = self.conv4(output)
        output = self.bn4(output)


        return output
class FCEncoder1(Module):
    def __init__(self, input_dim,  latent_size):
        super(FCEncoder1, self).__init__()
        self.block_channels = 64

        self.conv1 = conv16x1(1, self.block_channels, 2)
        self.bn1 = nn.BatchNorm1d(self.block_channels)

        self.max_pool = torch.nn.MaxPool1d(16, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        self.fem1 = FEM_RABlock(self.block_channels)
        self.fem2 = FEM_RABlock(self.block_channels * 2)
        self.fem3 = FEM_RABlock(self.block_channels * 4)

        self.fcm1 = FCM_ResBlock(self.block_channels * 8)
        self.fcm2 = FCM_ResBlock(self.block_channels * 2)

        self.glob_avg_pool = nn.AdaptiveMaxPool1d(1)
        self.glob_avg_pool1 = nn.AdaptiveMaxPool1d(2)
        self.bn2 = torch.nn.BatchNorm1d(int(self.block_channels/2))


        self.fc1 = conv1x1(int(self.block_channels/2), 32)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.fc2 = conv1x1(32, 16)
        self.fc_mu = conv1x1(16, 10)
        self.fc_logvar = conv1x1(16, 10)

        self.selu = torch.nn.SELU()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(dim=1)

        out = self.conv1(x)
        out = self.selu(out)
        out = self.bn1(out)

        out = self.max_pool(out)
        out = self.fem1(out)
        out = self.fem2(out)
        out = self.fem3(out)
        out = self.fcm1(out)
        out = self.fcm2(out)
        out = self.glob_avg_pool(out)
        out = self.bn2(out)

        out = self.fc1(out)
        out = self.selu(out)
        out = self.bn3(out)
        out = self.fc2(out)
        out = self.relu(out)

        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        mu = mu.squeeze(dim=2)
        logvar = logvar.squeeze(dim=2)


        # out = self.relu1(self.bn1(self.fc1(x)))
        # out = self.relu2(self.bn2(self.fc2(out)))
        # out = self.relu3(self.bn3(self.fc3(out)))
        # out = self.relu4(self.bn4(self.fc4(out)))
        # mu = self.fc_mu(out)
        # logvar = self.fc_logvar(out)


        return mu, logvar

class FCDecoder1(Module):
    def __init__(self, input_dim,latent_size):
        super(FCDecoder1, self).__init__()
        self.block_channels = 64
        self.conv1 = convT3x1(10,int(self.block_channels/4), 1)
        self.bn1 = nn.BatchNorm1d(int(self.block_channels/4))
        self.conv2 = convT3x1(int(self.block_channels/4),int(self.block_channels/2), 1)
        self.bn2 = nn.BatchNorm1d(int(self.block_channels/2))

        self.fc1 = nn.Linear(1, int(self.block_channels/4))
        self.bn3 = nn.BatchNorm1d(int(self.block_channels/2))
        self.fcm1 = FCM(int(self.block_channels/2))
        self.fcm2 = FCM(int(self.block_channels*2))
        self.fem1 = FEM(int(self.block_channels*8))
        self.fem2 = FEM(int(self.block_channels*4))
        self.fem3 = FEM(int(self.block_channels*2))
        self.fc2 = nn.Linear(121, input_dim)
        self.bn4 = nn.BatchNorm1d(int(self.block_channels))

        self.conv3 = conv1x1(self.block_channels,1 )
        self.bn5 = nn.BatchNorm1d(1)

        self.fc3 = nn.Linear(input_dim, input_dim)
        self.selu = torch.nn.SELU()
        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()

    def forward(self, z):
        z = z.unsqueeze(dim=2)
        out = self.conv1(z)
        out = self.selu(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.selu(out)
        out = self.bn2(out)
        out = self.fc1(out)
        out = self.selu(out)
        out = self.bn3(out)
        out = self.fcm1(out)
        out = self.fcm2(out)
        out = self.fem1(out)
        out = self.fem2(out)
        out = self.fem3(out)
        out = self.fc2(out)
        out = self.selu(out)
        out = self.bn4(out)
        out = self.conv3(out)
        out = self.selu(out)
        out = self.bn5(out)
        out = self.fc3(out)
        out = out.squeeze(dim=1)
        x_hat = self.sig(out)
        # x_hat = self.relu(out)
        return x_hat

class VAE(Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__()
        self.encoder = Encoder
        self.decoder = Decoder

    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)  # [batch_size,input_dim]
        z = self.reparameterization(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, z

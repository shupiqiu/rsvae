import torch
from torch import nn
from torch.nn import Module


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FCEncoder1(Module):
    def __init__(self, input_dim,  latent_size):
        super(FCEncoder1, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        # self.fc3 = nn.Linear(128, 64)
        # self.fc4 = nn.Linear(64, 32)
        self.fc_mu = nn.Linear(512, latent_size)
        self.fc_logvar = nn.Linear(512, latent_size)

        # self.relu1 = nn.ReLU()
        self.leakyrelu1 = nn.LeakyReLU()
        self.relu2 = nn.ReLU()
        # self.relu3 = nn.ReLU()
        # self.relu4 = nn.ReLU()

        # self.bn1 = torch.nn.BatchNorm1d(1)
        # self.bn2 = torch.nn.BatchNorm1d(1)
        # self.bn3 = torch.nn.BatchNorm1d(1)
        # self.bn4 = torch.nn.BatchNorm1d(1)

    def forward(self, x):
        out = self.leakyrelu1(self.fc1(x))
        out = self.relu2(self.fc2(out))
        # out = self.relu3(self.fc3(out))
        # out = self.relu4(self.fc4(out))
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
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
        self.fc1 = nn.Linear(latent_size, 128)
        self.fc2 = nn.Linear(128, 256)
        # self.fc3 = nn.Linear(64, 128)
        # self.fc4 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, input_dim)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.sig3 = nn.Sigmoid()
        # self.leakyrelu1 = nn.LeakyReLU()
        # self.leakyrelu2 = nn.LeakyReLU()
        # self.leakyrelu3 = nn.LeakyReLU()
        # self.leakyrelu4 = nn.LeakyReLU()
        # self.leakyrelu5 = nn.LeakyReLU()
        # self.bn1 = torch.nn.BatchNorm1d(1)
        # self.bn2 = torch.nn.BatchNorm1d(1)
        # self.bn3 = torch.nn.BatchNorm1d(1)
        # self.bn4 = torch.nn.BatchNorm1d(1)
        # self.bn5 = torch.nn.BatchNorm1d(1)

    def forward(self, z):
        out = self.relu1(self.fc1(z))
        out = self.relu2(self.fc2(out))
        # out = self.leakyrelu3(self.fc3(out))
        # out = self.leakyrelu4(self.fc4(out))
        x_hat = self.sig3(self.fc3(out))
        # out = self.relu1(self.bn1(self.fc1(z)))
        # out = self.relu2(self.bn2(self.fc2(out)))
        # out = self.relu3(self.bn3(self.fc3(out)))
        # out = self.relu4(self.bn4(self.fc4(out)))
        # x_hat = self.relu5(self.bn5(self.fc5(out)))

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

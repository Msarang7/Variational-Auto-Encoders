import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dims, device):
        super().__init__()

        self.device = device

        self.conv1 = nn.Conv2d(1,8,3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(8,16,3, stride = 2, padding = 1)
        self.batchNorm = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16,32,3, stride = 2, padding = 0)
        self.linear1 = nn.Linear(3*3*32, 128)
        self.linear2 = nn.Linear(128,latent_dims)
        self.linear3 = nn.Linear(128, latent_dims)

        self.N = torch.distributions.Normal(0,1)
        self.N.loc = self.N.loc.cuda() # easier to store N with cuda rather than changing others to cpu
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):

        x = x.to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.batchNorm(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim = 1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))

        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1).sum()
        return z



class Decoder(nn.Module):

    def __init__(self, latent_dims, device):
        super().__init__()

        self.device = device

        self.linear1 = nn.Linear(latent_dims, 128)
        self.linear2 = nn.Linear(128, 3*3*32)
        self.unflatten = nn.Unflatten(dim = 1, unflattened_size = (32,3,3))
        self.convT1 = nn.ConvTranspose2d(32,16,3, stride = 2, output_padding = 0)
        self.batchNorm1 = nn.BatchNorm2d(16)
        self.convT2 = nn.ConvTranspose2d(16,8, 3, stride = 2, padding = 1, output_padding = 1)
        self.batchNorm2 = nn.BatchNorm2d(8)
        self.convT3 = nn.ConvTranspose2d(8,1,3, stride = 2, padding = 1, output_padding = 1)

    def forward(self, x):

        x = x.to(self.device)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.unflatten(x)
        x = self.convT1(x)
        x = F.relu(self.batchNorm1(x))
        x = self.convT2(x)
        x = F.relu(self.batchNorm2(x))
        x = self.convT3(x)
        x = torch.sigmoid(x)
        return x

class VAE(nn.Module):

    def __init__(self, latent_dims, device):
        super(VAE, self).__init__()

        self.encoder = Encoder(latent_dims, device)
        self.decoder = Decoder(latent_dims, device)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        z = self.encoder(x)
        return self.decoder(z)





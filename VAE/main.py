import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms,datasets
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import warnings
from vae import VAE
from utils import train, test, plot_outputs, plot_latent_reconstruction, plot_encodings, loss_plots

warnings.filterwarnings('ignore')

##########################################################

data_dir = 'MNIST'

train_dataset = datasets.MNIST(data_dir, train = True, download = True)
test_dataset = datasets.MNIST(data_dir, train = False, download = True)


train_transforms = transforms.Compose([transforms.ToTensor()])
test_transforms = transforms.Compose([transforms.ToTensor()])

train_dataset.transform = train_transforms
test_dataset.transform = test_transforms

l = len(train_dataset)
train_data, val_data = random_split(train_dataset, [int(l - l*0.2), int(l*0.2)])
batch_size = 128

train_loader = DataLoader(train_data, batch_size = batch_size)
val_loader = DataLoader(val_data, batch_size = batch_size)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

#################################################uuu
torch.manual_seed(7)
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("gpu selected from training")
else :
    device = torch.device("cpu")
    print("cpu selected for training")

latent_dims = 2
vae = VAE(latent_dims, device)
vae.to(device)

lr = 1e-3
optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay = 1e-5)
epochs = 100


train_losses = []
val_losses = []
epoch_list = []

for epoch in range(epochs):

    # get losses

    train_loss = train(vae, train_loader, optim, device)
    val_loss = test(vae, val_loader, device)
    print("epoch : " + str(epoch) + "/" + str(epochs))
    print("train loss : " + str(train_loss))
    print("val loss : " + str(val_loss))

    # draw plots

    plot_outputs(vae.encoder, vae.decoder,  test_dataset, device, epoch, n = 5)
    plot_latent_reconstruction(vae.decoder, epoch, device)
    plot_encodings(vae.encoder, test_dataset, epoch, device)














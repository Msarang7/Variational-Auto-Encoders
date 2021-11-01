import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd

def train(vae, dataloader, optimizer, device):

    vae.train()
    train_loss = 0

    for x,_ in dataloader:

        x = x.to(device)

        x_hat = vae(x)
        loss = ((x - x_hat)**2).sum() + vae.encoder.kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print("batch loss = " + str(loss.item()))
        train_loss += loss.item()


    return train_loss / len(dataloader.dataset)

def test(vae, dataloader, device):

    vae.eval()
    val_loss = 0

    with torch.no_grad():

        for x,_ in dataloader :
            x = x.to(device)
            z = vae.encoder(x)
            x_hat = vae.decoder(z)

            loss = ((x-x_hat)**2).sum() + vae.encoder.kl
            val_loss += loss.item()
        return val_loss/len(dataloader.dataset)


def plot_outputs(encoder, decoder, test_dataset, device, epoch, n = 5):

    plt.figure(figsize = (10,5))

    for i in range(n):
        ax = plt.subplot(2,n,i+1) # num of rows, col, index
        img = test_dataset[i][0].unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            gen_img = decoder(encoder(img))

        # showing original image
        plt.imshow(img.cpu().squeeze().numpy(), cmap = 'gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n//2 :
            ax.set_title('original images')


        ax = plt.subplot(2,n, i+1+n)
        plt.imshow(gen_img.cpu().squeeze().numpy(), cmap = 'gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if i == n//2:
            ax.set_title('reconstruted images')
    plt.savefig('test_results/'+ 'epoch_' + str(epoch)+'.jpeg')

def plot_latent_reconstruction(decoder, epoch, device):

    plt.figure(figsize = (20,8))
    r0 = (-2,2)
    r1 = (-2,2)
    n = 10
    w = 28
    img = np.zeros((n*w, n*w))

    for i,y in enumerate(np.linspace(*r1,n)):
        for j,x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x,y]]).to(device)
            x_hat = decoder(z)
            x_hat = x_hat.reshape(28,28).to('cpu').detach().numpy()
            img[(n-1-i)*w : (n-1-i+1)*w , j*w : (j+1)*w] = x_hat

    plt.imshow(img,extent = [*r0, *r1], cmap = 'gist_gray')
    plt.savefig('reconstruction_results/'+ 'epoch_' + str(epoch)+'.jpeg')


# latent encodings for test dataset
def plot_encodings(encoder, test_dataset, epoch, device):

    encoded_samples = []
    for sample in test_dataset:
        img = sample[0].unsqueeze(0).to(device) # because input should be of size  a*b*c*d
        label = sample[1]

        encoder.eval()
        with torch.no_grad():
            encodings = encoder(img)

        encodings = encodings.flatten().cpu().numpy()
        encoded_sample = {f'enc var {i}' : enc for i, enc in enumerate(encodings)}
        encoded_sample['label'] = label
        encoded_samples.append(encoded_sample)
    encoded_samples = pd.DataFrame(encoded_samples)


    fig = px.scatter(encoded_samples, x='enc var 0', y='enc var 1', color=encoded_samples.label.astype(str),
               opacity=0.7)
    fig.write_image('encodings_Scatter_plot/' + 'epoch_' + str(epoch) + '.jpeg')


    #### tsne scatter plots ####

    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(encoded_samples.drop(['label'], axis = 1))
    fig = px.scatter(tsne_results, x=0, y=1, color=encoded_samples.label.astype(str),
                     labels={'0': 'tsne-2d-one', '1': 'tsne-2d-two'})
    fig.write_image('encodings_tsne/' + 'epoch_' + str(epoch) + '.jpeg')








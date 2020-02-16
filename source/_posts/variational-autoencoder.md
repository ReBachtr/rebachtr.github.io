---
title: variational_autoencoder
date: 2020-02-16 00:13:35
tags: Deeplearning
---

Reference: [Resource Paper](https://arxiv.org/pdf/1312.6114.pdf)
Variational AE is a kind of generative model, inspired by Autoencoders. It applied the variation inference to let the model learn the latent variables of the hidden distribution.
Here we use the MNIST dataset inside the torchvision.
### Data
```Python
import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_data = datasets.MNIST('../MNISTdata', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
test_data = datasets.MNIST('../MNISTdata', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
```
### Model
#### model 1 (MLP)
```Python
class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def encoder(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def decoder(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))
        
    def reparameterize(self, mu,logvar):
        std = logvar.mul(0.5).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
        
    def forward(self, x):
        mu, logvar = self.encoder(x.view(-1,784))
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
```
#### model 2 (CNN)
``` Python
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.hidden = 10
        self.en_conv_1 = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh()
        )
        self.en_conv_2 = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh()
        )
        self.en_fc_1 = nn.Linear(16 * 7 * 7, self.hidden)
        self.en_fc_2 = nn.Linear(16 * 7 * 7, self.hidden)
        self.de_fc = nn.Linear(self.hidden, 16 * 7 * 7)
        self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid()
        )
    def encoder(self, x):
        conv_out_1 = self.en_conv_1(x)
        conv_out_1 = conv_out_1.view(x.size(0), -1)
        conv_out_2 = self.en_conv_2(x)
        conv_out_2 = conv_out_2.view(x.size(0), -1)
        encoded_fc1 = self.en_fc_1(conv_out_1)
        encoded_fc2 = self.en_fc_2(conv_out_2)
        return encoded_fc1, encoded_fc2 

    def sampler(self, mean, std):
        var = std.mul(0.5).exp_()
        eps = torch.FloatTensor(var.size()).normal_()  
        eps = Variable(eps)
        eps = eps.to(device)
        return eps.mul(var).add_(mean)

    def decoder(self, x):
        out = self.de_fc(x)
        out = out.view(-1, 16, 7, 7)
        out = self.de_conv(out)
        return out

    def forward(self, x):
        mean, std = self.encoder(x)
        code = self.sampler(mean, std)
        out = self.decoder(code)
        return out, mean, std, code
```
### Loss function
``` Python
# reconstruction_function = nn.BCELoss()
reconstruction_function = nn.MSELoss()
reconstruction_function.size_average = False

def loss_function(recon_x, x, mu, logvar):
    simple_loss = reconstruction_function(recon_x.view(-1, 784), x.view(-1, 784))
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return simple_loss + 0.0002 * KLD
```
### Train
``` Python
model = VAE()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
model.train()
view_data = train_data.train_data[:5].view(-1,1,28,28).type(torch.cuda.FloatTensor)

EPOCH = 50
for epoch in range(EPOCH):
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, _ = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
            decoded_data, _, _, _ = model(view_data)
            f, a = plt.subplots(2, 5, figsize=(5, 2))
            for i in range(5):
                a[0][i].imshow(np.reshape((view_data.cpu()).data.numpy()[i], (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())
            for i in range(5):
                a[1][i].clear()
                a[1][i].imshow(np.reshape((decoded_data.cpu()).data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
            plt.show()
```
### Result
Here is the result using model 2, since model 1 has a poor performance. The upper images are original data, and the bottom images are generated by VAE model.

![epoch1.png](https://upload-images.jianshu.io/upload_images/18864424-285afecae0fe2835.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![epoch3.png](https://upload-images.jianshu.io/upload_images/18864424-2cf77d17460f6516.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![epoch5.png](https://upload-images.jianshu.io/upload_images/18864424-19a6c0a12f35e0fc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![epoch10.png](https://upload-images.jianshu.io/upload_images/18864424-3369749ad5faa840.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![epoch20.png](https://upload-images.jianshu.io/upload_images/18864424-7c49379c7b1680f3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![epoch30.png](https://upload-images.jianshu.io/upload_images/18864424-580109903524640d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![epoch40.png](https://upload-images.jianshu.io/upload_images/18864424-323bf96829d5fdaf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![epoch50.png](https://upload-images.jianshu.io/upload_images/18864424-afcd2550c22d290f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

To generate some more images, it is easy to use the **decoder** function of a trained model, by modifying the input of **sampler** (mean and std).
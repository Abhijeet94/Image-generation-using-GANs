



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.utils as vutils

from random import randint
from matplotlib import pyplot as plt
from IPython.display import Image
from IPython.core.display import Image, display

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


bs = 32




# Load Data
dataset = datasets.ImageFolder(root='/Users/jagtarsingh/OneDrive/UPenn/CIS680/VAE/cufs', transform=transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(), 
]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
print(len(dataset.imgs), len(dataloader))



fixed_x, _ = next(iter(dataloader))
save_image(fixed_x, 'real_image.png')

Image('real_image.png')



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)



class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)



class VAE(nn.Module):
    def __init__(self, channels=3, h_dim=1024, latent_dim=64):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, latent_dim)
        self.fc2 = nn.Linear(h_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        
    

    def forward(self, x):
        encoded_images =  self.encoder(x)
        mu, log_sig = self.fc1(encoded_images), self.fc2(encoded_images)
        std = log_sig.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        z = self.fc3(z)
        z = self.decoder(z)
        return z, mu, log_sig
    
    def decode_random(self):
        z = torch.randn([32,64])
        z = self.fc3(z)
        z = self.decoder(z)
        return z



channels = fixed_x.size(1)
print(channels)



model = VAE(channels=channels).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 



def loss_fn(recon_x, x, mu, log_sig):
    Recon_loss = F.binary_cross_entropy(recon_x, x, size_average=False)
#     Recon_loss = F.mse_loss(recon_x, x, size_average=False)

    KL_loss = -0.5 * torch.mean(1 + log_sig - mu.pow(2) - log_sig.exp())
#     KL_loss = 0
#     return Recon_loss
    return Recon_loss + KL_loss, Recon_loss, KL_loss



epochs = 60



itera = 0
loss_all = []
for epoch in range(epochs):
    for idx, (images, _) in enumerate(dataloader):
        recon_images, mu, log_sig = model(images)
        loss, Recon_loss, KL_loss = loss_fn(recon_images, images, mu, log_sig)
#         loss = loss_fn(recon_images, images, mu, log_sig)
        batch_s = images.size(0)
        loss_all.append(loss.data[0]/batch_s)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        itera+=1
        
        
        if batch_s == 23:
            n = min(images.size(0), 16)
            fixed_out,_,_ = model(fixed_x)
            comp = torch.cat([fixed_x[:n],fixed_out[:n]])
            comparison = torch.cat([images[:n], recon_images[:n]])
            save_image(comp.data.cpu(),
                         './reconstructed/reconstruction_' + str(epoch) + '.png', nrow=n)
            print("Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1, 
                                epochs, loss.data[0]/bs, Recon_loss.data[0]/bs, KL_loss.data[0]/bs))
            random_out_img = model.decode_random()
            
            save_image(random_out_img.data.cpu(),
                         './reconstructed_rand/random_' + str(epoch) + '.png', nrow=n)
            
            

plt.plot(loss_all)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Vs Iteration for CUFS dataset')
plt.savefig('cufs_loss.png')
torch.save(model.state_dict(), 'vae.torch')

recon_images, _, _ = model(fixed_x)
comparison = torch.cat([fixed_x[:],recon_images[:]])
save_image(comparison.data.cpu(),
                         './reconstructed/reconstruction_last' + str(epoch) + '.png', nrow=n)


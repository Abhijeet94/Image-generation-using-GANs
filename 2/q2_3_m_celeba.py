# -*- coding: utf-8 -*-

import pdb
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from networks import *

class DCGAN:
	def __init__(self, dataroot):
		manualSeed = 1
		#manualSeed = random.randint(1, 10000)
		print("Random Seed: ", manualSeed)
		random.seed(manualSeed)
		torch.manual_seed(manualSeed)

		self.dataroot = dataroot
		self.batch_size = 128
		self.image_size = 64
		self.nc = 1
		self.num_epochs = 7
		# self.lr = 0.005
		self.lrG = 0.00005
		self.lrD = 0.00005
		self.nz = 100 # noise dimension
		self.ngf = 64 # size of generator feature map
		self.ndf = 64 # size of discriminator feature map
		self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
		self.criterion = nn.BCELoss()

		self.d_iter = 5
		self.clamp_lower = -0.01
		self.clamp_upper = 0.01

	def loadData(self):
		print ("Making dataset")
		dataset = dset.ImageFolder(root=self.dataroot,
								   transform=transforms.Compose([
									   transforms.Resize(self.image_size),
									   transforms.Grayscale(1),
									   transforms.CenterCrop(self.image_size),
									   transforms.ToTensor(),
									   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
								   ]))
		self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)


	def getGenerator(self):
		netG = Generator(1, self.nz, self.ngf, self.nc).to(self.device)
		netG.apply(weights_init)
		return netG

	def getDiscriminator(self):
		netD = Discriminator(1, self.ndf, self.nc).to(self.device)
		netD.apply(weights_init)
		return netD

	def train(self):
		netG = self.getGenerator()
		netD = self.getDiscriminator()
		optimizerD = optim.RMSprop(netD.parameters(), lr=self.lrD)
		optimizerG = optim.RMSprop(netG.parameters(), lr=self.lrG)
		testNoise = torch.randn(64, self.nz, 1, 1, device=self.device)

		self.intermediateImages = []
		self.GeneratorLoss = []
		self.DiscriminatorLoss = []
		iters = 0
		realImageLabel = 1
		fakeImageLabel = 0

		print("Starting Training Loop...")
		for epoch in range(self.num_epochs):

			i = 0
			data_iter = iter(self.dataloader)
			while i < len(self.dataloader):
				
				#######################################################
				# Train Discriminator

				for p in netD.parameters(): # reset requires_grad
					p.requires_grad = True # they are set to False below in netG update

				j = 0
				while j < self.d_iter and i < len(self.dataloader):
					j += 1
					data = data_iter.next()
					i += 1

					for p in netD.parameters():
						p.data.clamp_(self.clamp_lower, self.clamp_upper)

					# Real batch
					netD.zero_grad()
					realData = data[0].to(self.device)
					b_size = realData.size(0)
					label = torch.full((b_size,), realImageLabel, device=self.device)
					output = netD(realData).view(-1)
					errorDiscriminatorReal = output.mean() #self.criterion(output, label)
					errorDiscriminatorReal.backward()
					D_x_out = output.mean().item()

				# Fake batch
				noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
				fake = netG(noise)
				label.fill_(fakeImageLabel)
				output = netD(fake.detach()).view(-1)
				errorDiscriminatorFake = -1.0 * output.mean() #self.criterion(output, label)
				errorDiscriminatorFake.backward()
				D_G_z1_out = output.mean().item()
				errD = errorDiscriminatorReal + errorDiscriminatorFake

				optimizerD.step()

				#######################################################
				# Train Generator

				for p in netD.parameters():
					p.requires_grad = False # to avoid computation

				netG.zero_grad()
				label.fill_(realImageLabel)
				output = netD(fake).view(-1)
				errorGenerator = output.mean() #self.criterion(output, label)
				errorGenerator.backward()
				D_G_z2_out = output.mean().item()

				optimizerG.step()

				#######################################################
				# Print and record training statistics

				if (i - 1) % 50 == self.d_iter - 1 or (i) % 50 == self.d_iter - 1 or (i + 1) % 50 == self.d_iter - 1:
					print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
						  % (epoch, self.num_epochs, i, len(self.dataloader),
							 errD.item(), errorGenerator.item(), D_x_out, D_G_z1_out, D_G_z2_out))
				
				self.GeneratorLoss.append(errorGenerator.item())
				self.DiscriminatorLoss.append(errD.item())
				
				if (iters % 500 == 0) or ((epoch == self.num_epochs-1) and (i-1 == len(self.dataloader)-1)):
					with torch.no_grad():
						fake = netG(testNoise).detach().cpu()
					self.intermediateImages.append(vutils.make_grid(fake, padding=2, normalize=True))
					# plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True),(1,2,0)))
					# plt.show()
					
				iters += 1

	def plotResults(self):
		plt.figure(figsize=(10,5))
		plt.title("Training loss - Generator and Discriminator")
		plt.plot(self.GeneratorLoss,label="Generator")
		plt.plot(self.DiscriminatorLoss,label="Discriminator")
		plt.xlabel("# iterations")
		plt.ylabel("Loss")
		plt.legend()
		plt.show()


		for i, image in enumerate(self.intermediateImages):
			plt.figure(figsize=(10,5))
			plt.title("Generator's output during training #" + str(i + 1) + "/" + str(len(self.intermediateImages)))
			plt.axis("off")
			plt.imshow(np.transpose(image, (1,2,0)))
			plt.show()


		real_batch = next(iter(self.dataloader)) # next real batch
		plt.figure(figsize=(15,15))
		plt.subplot(1,2,1)
		plt.axis("off")
		plt.title("Real Images")
		plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

		plt.subplot(1,2,2) # fake images
		plt.axis("off")
		plt.title("Fake Images")
		plt.imshow(np.transpose(self.intermediateImages[-1],(1,2,0)))
		plt.show()


if __name__ == '__main__':
	dcgan = DCGAN("celeba")
	dcgan.loadData()
	dcgan.train()
	dcgan.plotResults()
	print ('Done')
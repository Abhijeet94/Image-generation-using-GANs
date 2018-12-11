# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import pdb
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# The Generator and Discriminator are constructed for image size 64
# i.e, the generator generates image size of 64 X 64 and the discriminator takes
# input of image size 64 X 64

class Generator(nn.Module):
	def __init__(self, ngpu, nz, ngf, nc):
		super(Generator, self).__init__()
		self.ngpu = ngpu

		self.block1 = nn.Sequential(
			# input is nz, random noise
			nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True)
			)

		self.block2 = nn.Sequential(
			# convolution layer size: (ngf*8) x 4 x 4
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.ReLU(True)
			)

		self.block3 = nn.Sequential(
			# convolution layer size: (ngf*4) x 8 x 8
			nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 2),
			nn.ReLU(True)
			)

		self.block4 = nn.Sequential(
			# convolution layer size: (ngf*2) x 16 x 16
			nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf),
			nn.ReLU(True)
			)

		self.out = nn.Sequential(
			# convolution layer size: (ngf) x 32 x 32
			nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
			nn.Tanh()
			# output size. (nc) x 64 x 64
			)

	def forward(self, input):
		output = self.block1(input)
		output = self.block2(output)
		output = self.block3(output)
		output = self.block4(output)
		output = self.out(output)
		return output


class Discriminator(nn.Module):
	def __init__(self, ngpu, ndf, nc):
		super(Discriminator, self).__init__()
		self.ngpu = ngpu

		self.block1 = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True)
			)

		self.block2 = nn.Sequential(
			# convolution layer size: (ndf) x 32 x 32
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True)
			)

		self.block3 = nn.Sequential(
			# convolution layer size: (ndf*2) x 16 x 16
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True)
			)

		self.block4 = nn.Sequential(
			# convolution layer size: (ndf*4) x 8 x 8
			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True)
			)

		self.out = nn.Sequential(
			# convolution layer size: (ndf*8) x 4 x 4
			nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
			)

	def forward(self, input):
		output = self.block1(input)
		output = self.block2(output)
		output = self.block3(output)
		output = self.block4(output)
		output = self.out(output)
		return output


# custom weights initialization called on netG and netD
def weights_init(model):
	classname = model.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(model.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(model.weight.data, 1.0, 0.02)
		nn.init.constant_(model.bias.data, 0)


class DCGAN:
	def __init__(self, dataroot, batchSize=128, imageSize=64, nc=1, numEpochs=500, lr=0.0002, nz=100, ngf=64, ndf=64):
		manualSeed = 1
		#manualSeed = random.randint(1, 10000)
		print("Random Seed: ", manualSeed)
		random.seed(manualSeed)
		torch.manual_seed(manualSeed)

		self.dataroot = dataroot
		self.batch_size = batchSize
		self.image_size = imageSize
		self.nc = nc
		self.num_epochs = numEpochs
		self.lr = lr
		self.nz = nz # noise dimension
		self.ngf = ngf # size of generator feature map
		self.ndf = ndf # size of discriminator feature map
		self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
		self.criterion = nn.BCELoss()

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
		optimizerD = optim.Adam(netD.parameters(), lr=self.lr, betas=(0.5, 0.999))
		optimizerG = optim.Adam(netG.parameters(), lr=self.lr, betas=(0.5, 0.999))
		testNoise = torch.randn(64, self.nz, 1, 1, device=self.device)

		self.intermediateImages = []
		self.GeneratorLoss = []
		self.DiscriminatorLoss = []
		iters = 0
		realImageLabel = 1
		fakeImageLabel = 0

		print("Starting Training Loop...")
		for epoch in range(self.num_epochs):

			for i, data in enumerate(self.dataloader, 0):
				
				#######################################################
				# Train Discriminator

				# Real batch
				netD.zero_grad()
				realData = data[0].to(self.device)
				b_size = realData.size(0)
				label = torch.full((b_size,), realImageLabel, device=self.device)
				output = netD(realData).view(-1)
				errorDiscriminatorReal = self.criterion(output, label)
				errorDiscriminatorReal.backward()
				D_x_out = output.mean().item()

				# Fake batch
				noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
				fake = netG(noise)
				label.fill_(fakeImageLabel)
				output = netD(fake.detach()).view(-1)
				errorDiscriminatorFake = self.criterion(output, label)
				errorDiscriminatorFake.backward()
				D_G_z1_out = output.mean().item()
				errD = errorDiscriminatorReal + errorDiscriminatorFake

				optimizerD.step()

				#######################################################
				# Train Generator

				netG.zero_grad()
				label.fill_(realImageLabel)
				output = netD(fake).view(-1)
				errorGenerator = self.criterion(output, label)
				errorGenerator.backward()
				D_G_z2_out = output.mean().item()

				optimizerG.step()

				#######################################################
				# Print and record training statistics

				if i % 50 == 0:
					print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
						  % (epoch, self.num_epochs, i, len(self.dataloader),
							 errD.item(), errorGenerator.item(), D_x_out, D_G_z1_out, D_G_z2_out))
				
				self.GeneratorLoss.append(errorGenerator.item())
				self.DiscriminatorLoss.append(errD.item())
				
				if (iters % 500 == 0) or ((epoch == self.num_epochs-1) and (i == len(self.dataloader)-1)):
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


class WGAN:
	def __init__(self, dataroot, batchSize=128, imageSize=64, nc=1, numEpochs=500, lrG=0.0005, lrD=0.0005, nz=100, ngf=64, ndf=64, d_iter=3):
		manualSeed = 1
		#manualSeed = random.randint(1, 10000)
		print("Random Seed: ", manualSeed)
		random.seed(manualSeed)
		torch.manual_seed(manualSeed)

		self.dataroot = dataroot
		self.batch_size = batchSize
		self.image_size = imageSize
		self.nc = nc
		self.num_epochs = numEpochs
		# self.lr = 0.005
		self.lrG = lrG
		self.lrD = lrD
		self.nz = nz # noise dimension
		self.ngf = ngf # size of generator feature map
		self.ndf = ndf # size of discriminator feature map
		self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
		self.criterion = nn.BCELoss()

		self.d_iter = d_iter
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
import torch
import torch.nn as nn
import numpy as np

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
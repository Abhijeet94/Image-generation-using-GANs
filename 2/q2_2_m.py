# -*- coding: utf-8 -*-

from networks import *

if __name__ == '__main__':
	dcgan = DCGAN(dataroot="celeba", batchSize=128, imageSize=64, nc=1, numEpochs=5, lr=0.0002, nz=100, ngf=64, ndf=64)
	dcgan.loadData()
	dcgan.train()
	dcgan.plotResults()
	print ('Done')
# -*- coding: utf-8 -*-

from networks import *

if __name__ == '__main__':
	wgan = WGAN(dataroot="celeba", batchSize=128, imageSize=64, nc=1, numEpochs=7, lrG=0.00005, lrD=0.00005, nz=100, ngf=64, ndf=64, d_iter=5)
	wgan.loadData()
	wgan.train()
	wgan.plotResults()
	print ('Done')
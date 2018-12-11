# -*- coding: utf-8 -*-

from networks import *

if __name__ == '__main__':
	wgan = WGAN(dataroot="cufs2", batchSize=128, imageSize=64, nc=1, numEpochs=1200, lrG=0.0005, lrD=0.0005, nz=100, ngf=16, ndf=16, d_iter=3)
	wgan.loadData()
	wgan.train()
	wgan.plotResults()
	print ('Done')
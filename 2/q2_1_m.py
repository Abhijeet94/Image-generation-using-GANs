# -*- coding: utf-8 -*-

from networks import *

if __name__ == '__main__':
	dcgan = DCGAN(dataroot="cufs2", batchSize=128, imageSize=64, nc=1, numEpochs=400, lr=0.005, nz=100, ngf=16, ndf=16)
	dcgan.loadData()
	dcgan.train()
	dcgan.plotResults()
	print ('Done')

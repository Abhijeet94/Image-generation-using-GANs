# -*- coding: utf-8 -*-
"""hw3_new_new.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10nSSRiJKNFSo-BgFZ8R5qCXKntLguKBT
"""

# from google.colab import drive
# drive.mount('/content/gdrive/')

# from os.path import exists
# from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
# platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
# cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
# accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

# !pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.1-{platform}-linux_x86_64.whl torchvision
# import torch
# device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# !pip install Pillow==4.0.0
# !pip install PIL
# !pip install image

import os, time, pickle, argparse, pdb, random, time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torch.utils.data as data
from PIL import Image
import torch.nn.functional as F
from scipy.misc import imresize
import numpy as np
import matplotlib.pyplot as plt

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir)

    for root, _, basenames in sorted(os.walk(dir)):
        for basename in sorted(basenames):
            if basename.endswith(".jpg"):
                path = os.path.join(root, basename)
                images.append(path)

    return images


class ImageLoader(data.Dataset):

    def __init__(self, rootA, rootB, transform=None):
        imgsA = make_dataset(rootA)
        assert len(imgsA) > 0
        self.imgsA = imgsA

        imgsB = make_dataset(rootB)
        assert len(imgsB) > 0
        self.imgsB = imgsB

        self.transform = transform

    def __getitem__(self, index):
        seed = np.random.randint(time.time())

        pathA = self.imgsA[index]
        imgA = Image.open(pathA).convert("RGB")
        random.seed(seed)
        if self.transform is not None:
            imgA = self.transform(imgA)

        pathB = self.imgsB[index]
        imgB = Image.open(pathB).convert("RGB")
        random.seed(seed)
        if self.transform is not None:
            imgB = self.transform(imgB)

        return imgA, imgB

    def __len__(self):
        return len(self.imgsA)

# data_loader
transformTrain = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

transformTest = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

image_location = "CUHK"
batch_size = 1
shuffle = True
numTrainEpochs = 50
imgSize = 200
ngf = 64
ndf = 64
lrG = 0.0002
lrD = 0.0002
inverse_order = True
L1_lambda = 100
root = image_location + "_results_testOurs_P2S/"
model = "CUHK_"

train_dataset = ImageLoader(os.path.join(image_location, "trainA"), os.path.join(image_location, "trainB"), transformTrain)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

test_dataset = ImageLoader(os.path.join(image_location, "testOurs"), os.path.join(image_location, "testOurs"), transformTest)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

def show_result(G, x_, y_, num_epoch, show = False, save = False, path = 'result.png'):
    # G.eval()
    test_images = G(x_)

    size_figure_grid = 3
    fig, ax = plt.subplots(x_.size()[0], size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(x_.size()[0]), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for i in range(x_.size()[0]):
        ax[i, 0].cla()
        ax[i, 0].imshow((x_[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        ax[i, 1].cla()
        ax[i, 1].imshow((test_images[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        ax[i, 2].cla()
        ax[i, 2].imshow((y_[i].numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

class generator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(generator, self).__init__()
        # Unet encoder
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv6_bn = nn.BatchNorm2d(d * 8)
        self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv7_bn = nn.BatchNorm2d(d * 8)
        self.conv8 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        # self.conv8_bn = nn.BatchNorm2d(d * 8)

        # Unet decoder
        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d * 8)
        self.deconv5 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(d * 4)
        self.deconv6 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)
        self.deconv6_bn = nn.BatchNorm2d(d * 2)
        self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
        self.deconv7_bn = nn.BatchNorm2d(d)
        self.deconv8 = nn.ConvTranspose2d(d * 2, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        encoder1 = self.conv1(input)
        encoder2 = self.conv2_bn(self.conv2(F.leaky_relu(encoder1, 0.2)))
        encoder3 = self.conv3_bn(self.conv3(F.leaky_relu(encoder2, 0.2)))
        encoder4 = self.conv4_bn(self.conv4(F.leaky_relu(encoder3, 0.2)))
        encoder5 = self.conv5_bn(self.conv5(F.leaky_relu(encoder4, 0.2)))
        encoder6 = self.conv6_bn(self.conv6(F.leaky_relu(encoder5, 0.2)))
        encoder7 = self.conv7_bn(self.conv7(F.leaky_relu(encoder6, 0.2)))
        encoder8 = self.conv8(F.leaky_relu(encoder7, 0.2))
        decoder1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(encoder8))), 0.5, training=True)
        decoder1 = torch.cat([decoder1, encoder7], 1)
        decoder2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(decoder1))), 0.5, training=True)
        decoder2 = torch.cat([decoder2, encoder6], 1)
        decoder3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(decoder2))), 0.5, training=True)
        decoder3 = torch.cat([decoder3, encoder5], 1)
        decoder4 = self.deconv4_bn(self.deconv4(F.relu(decoder3)))
        decoder4 = torch.cat([decoder4, encoder4], 1)
        decoder5 = self.deconv5_bn(self.deconv5(F.relu(decoder4)))
        decoder5 = torch.cat([decoder5, encoder3], 1)
        decoder6 = self.deconv6_bn(self.deconv6(F.relu(decoder5)))
        decoder6 = torch.cat([decoder6, encoder2], 1)
        decoder7 = self.deconv7_bn(self.deconv7(F.relu(decoder6)))
        decoder7 = torch.cat([decoder7, encoder1], 1)
        decoder8 = self.deconv8(F.relu(decoder7))
        output = torch.tanh(decoder8)

        return output

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(6, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


G = generator(ngf)
G
G.load_state_dict(torch.load('generator_param2.pkl'))
G.eval()

if not os.path.isdir(image_location + '_results_testOurs_P2S'):
    os.mkdir(image_location + '_results_testOurs_P2S')

# network
with torch.no_grad():
    n = 0
    for x_, y_ in test_loader:
        if inverse_order:
            x_, y_ = y_, x_

        x_ = Variable(x_)
        test_image = G(x_)

        path = image_location + '_results_testOurs_P2S/' + str(n) + '_input.png'
        plt.imsave(path, (x_[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        path = image_location + '_results_testOurs_P2S/' + str(n) + '_output.png'
        plt.imsave(path, (test_image[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        path = image_location + '_results_testOurs_P2S/' + str(n) + '_target.png'
        plt.imsave(path, (y_[0].numpy().transpose(1, 2, 0) + 1) / 2)

        n += 1
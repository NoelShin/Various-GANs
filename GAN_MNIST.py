import os  # import os

import torchvision.transforms as transforms  # import transforms module for data preprocessing
from torchvision.utils import save_image  # import save_image module for saving image

from torch.utils.data import DataLoader  # import DataLoader class to load data
from torchvision import datasets  # import dataset class to make own datasets

import torch  # import pytorch
import torch.nn as nn  # import nn module for building neural networks

import numpy as np
from PIL import Image

N_EPOCHS = 200
N_DISC = 1
BATCH_SIZE = 64
lr = 0.0002
BETA_1 = 0.5
BETA_2 = 0.9
N_WORKERS = 2
DIM_LATENT = 100
IMG_SIZE = 28 # MNIST image size
INPUT_CH = OUTPUT_CH = 1
SAVE_INTERVAL = 10000

USE_CUDA = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.set_weight()

    def set_weight(self):
        model = []

        self.l1 = nn.Linear(DIM_LATENT, 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, 512)
        self.l4 = nn.Linear(512, 1024)
        self.l5 = nn.Linear(1024, OUTPUT_CH*IMG_SIZE*IMG_SIZE)

        self.act = nn.LeakyReLU(0.2, inplace=True)

        model += [self.l1, self.act, self.l2, self.act, self.l3, self.act, self.l4, self.act, self.l5, nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, z):
        x = self.model(z)
        x = x.view(BATCH_SIZE, INPUT_CH, IMG_SIZE, IMG_SIZE)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.set_weight()

    def set_weight(self):
        model = []

        self.l1 = nn.Linear(INPUT_CH*IMG_SIZE*IMG_SIZE, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 1)

        self.act = nn.LeakyReLU(0.2, inplace=True)

        model += [self.l1, self.act, self.l2, self.act, self.l3, nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = x.view(BATCH_SIZE, -1)
        validity = self.model(x)

        return validity

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0]))/(np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0])*scale)
        data = data*scale + bias
    return data

def tensor2image(image_tensor):
    np_image = image_tensor.squeeze().cpu().float().numpy()
    # assert np_image.shape[0] in [1, 3], print("The channel is ", np_image.shape)
    if len(np_image.shape) == 2:
        pass
    elif len(np_image.shape) == 3:
        np_image = np.transpose(np_image, (1, 2, 0))

    np_image = adjust_dynamic_range(np_image, drange_in=[-1., 1.], drange_out=[0, 255])
    np_image = np.clip(np_image, 0, 255).astype(np.uint8)
    return np_image

def save_image(image_tensor, path, mode='png'):
    np_image = tensor2image(image_tensor)
    pil_image = Image.fromarray(np_image)
    pil_image.save(path, mode)

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    dataset = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True,
                                              num_workers=N_WORKERS)

    G = Generator()
    D = Discriminator()

    if USE_CUDA:
        G = G.cuda()
        D = D.cuda()

    GAN_Loss = nn.BCELoss()

    G_optim = torch.optim.Adam(G.parameters(), lr=lr, betas=(BETA_1, BETA_2))
    D_optim = torch.optim.Adam(D.parameters(), lr=lr, betas=(BETA_1, BETA_2))

    total_step = 0
    for epoch in range(N_EPOCHS):
        for i, (targets, _) in enumerate(data_loader):
            total_step += 1
            valid = torch.Tensor(BATCH_SIZE, 1).fill_(1.0)
            fake = torch.Tensor(BATCH_SIZE, 1).fill_(0.0)

            if USE_CUDA:
                valid.cuda()
                fake.cuda()

            reals = torch.tensor(targets, requires_grad=False)
            z = torch.randn(BATCH_SIZE, DIM_LATENT) # make a random normal noise

            if USE_CUDA:
                reals.cuda()
                z.cuda()

            fakes = G(z) # generate fake samples

            real_loss = GAN_Loss(D(reals), valid)
            fake_loss = GAN_Loss(D(fakes.detach()), fake)

            D_loss = (real_loss + fake_loss)*0.5

            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()

            if total_step % N_DISC == 0:
                z = torch.randn(BATCH_SIZE, DIM_LATENT)
                fakes = G(z)

                G_loss = GAN_Loss(D(fakes), valid)

                G_optim.zero_grad()
                G_loss.backward()
                G_optim.step()
                print(D_loss.detach().mean().squeeze().numpy(),
                              G_loss.item())
                #print("Epoch: [{}/{}] Batch: [{}/{}], D_Loss: {:.{prec}}, G_Loss: {:.{prec}}"
                #      .format(epoch, N_EPOCHS,
                #              total_step,
                #              BATCH_SIZE*(i+1), len(data_loader),
                #              D_loss.detach().mean().squeeze(),
                #              G_loss.item(),
                #              prec=4))

            if total_step % SAVE_INTERVAL:
                save_image(fakes[0].detach(), path=os.path.join('/Users/noel/Projects/Python/ML_class_practice/Lab7',
                                                                str(SAVE_INTERVAL)))

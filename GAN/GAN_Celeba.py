import os  # import os
import glob # import glob module for loading data
import torchvision.transforms as transforms  # import transforms module for data preprocessing

from torch.utils.data import DataLoader  # import DataLoader class to load data
from torchvision import datasets  # import dataset class to make own datasets

import torch  # import pytorch
import torch.nn as nn  # import nn module for building neural networks

import numpy as np # import numpy package for matrix calculation
from PIL import Image  # import Image module for saving image
import matplotlib.pyplot as plt # import matplotlib.pyplot for plotting loss graph

import datetime  # import datetime module for measuring time taken to train your model

N_EPOCHS = 1000  # the number of the epoch yo want
N_DISC = 1  # how many discriminator updates you want before a update for the generator
BATCH_SIZE = 64  # how many batchsize you want to use during learning
lr = 0.0002  # learning rate
BETA_1 = 0.5  # hyperparameter for Adam solver. Coefficient used for computing running averages of gradient
BETA_2 = 0.9  # hyperparameter for Adam solver. Coefficient used for computing running averages of square of gradient
N_WORKERS = 2  # how many CPU threads you want to use for loading data
DIM_LATENT = 10  # how many latent dimensions you want to set
IMG_SIZE = 28  # MNIST image size, i.e. a width(or height as it has square shape) of input and output data.
INPUT_CH = OUTPUT_CH = 3  # MNIST data is gray image which has only one channel
SAVE_INTERVAL = 10000  # how frequent you want to save the model's image

USE_CUDA = True if torch.cuda.is_available() else False  # If you have CUDA device(GPU), you can use it for reducing
# learning time


class Generator(nn.Module):  # define a generator class which inherites nn.Module class
    def __init__(self):  # define a constructor
        super(Generator, self).__init__()  # ensure the MRO rule
        self.set_weight()  # set weight layers for your model
        self.build_model()  # build your model after you set the weight layers

    def set_weight(self):  # define a method for setting weight layers
        self.l1 = nn.Linear(DIM_LATENT, 128)  # the first linear layer from latent dimension
        self.l2 = nn.Linear(128, 256)  # the second linear layer
        self.l3 = nn.Linear(256, 512)  # the third linear layer
        self.l4 = nn.Linear(512, 1024)  # the fourth linear layer
        self.l5 = nn.Linear(1024, OUTPUT_CH*IMG_SIZE*IMG_SIZE)  # the last linear layer

    def build_model(self):  # define a method for building your model
        def block(weight, normalize=True, activation=True):  # define a function for a repetitive block.
            layers = []  # an empty list for carrying layers
            layers += [weight]  # add a weight layer
            if normalize:  # if you want to add normalization
                norm = nn.BatchNorm1d(weight.weight.shape[0], momentum=0.1)  # define a Batch normalization layer
                layers += [norm]  # add the batch normalization layer
            if activation:  # if you want to add activation layer
                layers += [nn.LeakyReLU(0.2, inplace=True)]  # add an activation

            return layers  # return the block

        model = []  # an empty list for carrying blocks
        model += block(self.l1)  # add the first block
        model += block(self.l2)  # add the second block
        model += block(self.l3)  # add the third block
        model += block(self.l4)  # add the fourth block
        model += [*block(self.l5, activation=False), nn.Tanh()]  # add the last block with Tanh activation

        self.model = nn.Sequential(*model)  # convert the python list into nn.Sequential list. This should be done for
        # the parameters() method to recognize weight layers defined in the class

    def forward(self, z):  # method overriding of forward. This will be implemented when an instance of this class is called
        x = self.model(z)  # produce the output
        x = x.view(-1, INPUT_CH, IMG_SIZE, IMG_SIZE)  # transform the shape to MNIST shape
        return x  # return the output image


class Discriminator(nn.Module):  # define a discriminator class which inherites nn.Module class
    def __init__(self):   # define a constructor
        super(Discriminator, self).__init__()  # ensure the MRO rule
        self.set_weight()  # set weight layers for your model
        self.build_model()  # build your model after you set the weight layers

    def set_weight(self):   # define a method for setting weight layers
        self.l1 = nn.Linear(INPUT_CH*IMG_SIZE*IMG_SIZE, 512)  # define the first linear layer
        self.l2 = nn.Linear(512, 256)  # define the second linear layer
        self.l3 = nn.Linear(256, 1)  # define the third linear layer

    def build_model(self):  # define a method for building your model
        act = nn.LeakyReLU(0.2, inplace=True)  # define your activation
        model = [self.l1, act, self.l2, act, self.l3, nn.Sigmoid()]  # define your model. Note that we do not use
        # batch normalization for discriminator. If you add that, it makes the training harder. Don't know why yet.
        # If you know the reason, plz comment that for me. Thanks.

        self.model = nn.Sequential(*model)  # convert the python list into nn.Sequential list. This should be done for
        # the parameters() method to recognize weight layers defined in the class

    def forward(self, x):  # method overriding of forward. This will be implemented when an instance of this class is called
        x = x.view(-1, INPUT_CH*IMG_SIZE*IMG_SIZE)  # flatten the input except for the batch dimension
        validity = self.model(x)  # discriminate how possibly the input data is to real data
        return validity  # return the validity


def adjust_dynamic_range(data, drange_in, drange_out):  # define a function for scaling a data
    if drange_in != drange_out:  # if the range of pixel values are different for input and output
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0]))/(np.float32(drange_in[1]) - np.float32(drange_in[0]))
        # calculate a scaling factor
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0])*scale)
        # calculate a bias
        data = data*scale + bias
        # change the input data based on the scalining factor and bias
    return data # return the scaled data whose pixel values are within drange_out


def tensor2image(image_tensor):  # define a function for changing torch.tensor to a numpy array before saving image
    np_image = image_tensor.squeeze().cpu().float().numpy()
    # squeeze a input tensor (which means to delete dimensions with value 1) and convert it to cpu tensor(this is for
    # the case you use GPU during training). Ensure the pixel values have float type and finally convert to numpy array.
    if len(np_image.shape) == 2:  # if the array has only two dimensions (which means it is gray image)
        pass  # pass without changing the order of the axes
    elif len(np_image.shape) == 3:  # if the array has three dimensions (which means t is color(RGB) image)
        np_image = np.transpose(np_image, (1, 2, 0))  # change the order of the axes from (C, H, W) to (H, W, C)

    np_image = adjust_dynamic_range(np_image, drange_in=[-1., 1.], drange_out=[0, 255])  # scale the pixel values
    np_image = np.clip(np_image, 0, 255).astype(np.uint8)  # make its type uint8 so that you can save the image
    return np_image  # return the processed image


def save_image(image_tensor, path, mode='png'):  # define a function for saving processed image
    np_image = tensor2image(image_tensor)  # change a torch.tensor to a numpy image
    pil_image = Image.fromarray(np_image)  # convert the numpy image to Image object
    pil_image.save(path + '.png', mode)  # save the image with given path and mode


class Custom_dataset(torch.utils.data.Dataset):  # define a class for making own dataset
    def __init__(self, data_dir, format='png'):  # define a constructor
        super(Custom_dataset, self).__init__()  # ensure the MRO rule
        self.data_path_list = glob.glob(os.path.join(data_dir, '*.' + format))  # data path list

    def get_transform(self, normalize=True):  # define a function for getting transformation
        transform_list = []  # an emply list for carrying transforms
        transform_list += [transforms.ToTensor()]  # add a transform for converting image data to torch.tensor object

        if normalize:  # if you want normalize
            transform_list += [transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]  # normalize the
            # tensor object to have pixel values within [-1, 1]

        return transforms.Compose(transform_list)  # return transform list

    def __getitem__(self, index):  # method overriding for getting item from the Custom_dataset
        data_path = self.data_path_list[index]  # data path
        data = Image.open(data_path)  # open the path as Image object
        transform = self.get_transform()  # get transform listS
        data_tensor = transform(data)  # transform the Image object

        return data_tensor  # return the transformed tensor objects

    def __len__(self):  # method overriding for letting you know how big the dataset is

        return len(self.data_path_list)  # return the number of the dataset


if __name__ == '__main__':  # if this script is directly implemented(i.e., not imported), do the below codes
    start_time = datetime.datetime.now()  # count a start time
    dir = os.path.dirname(__file__)  # directory where this file is
    data_dir = os.path.join(dir, 'Celeba_10000')  # directory where the data is

    dataset = Custom_dataset(data_dir)  # define an instance of the Custom_dataset class

    # define dataset and designate what transforms you want during the loading. If you want to download the data, you
    # can set the download keyword to True. This is not necessary.
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True,
                                              num_workers=N_WORKERS)
    # define dataloader and designate the batch_size you want. Setting the shuffle keyword True shuffle the order of the
    # dataset after every epoch. num_workers keywords means how many CPU threads you want to use for loading the data.
    G = Generator()  # define an instance of the Generator class
    D = Discriminator()  # define an instance of the Discriminator class

    if USE_CUDA:  # if you have GPU device
        G = G.cuda()  # set your model (specifically the weights(parameters) defined in the model) in GPU
        D = D.cuda()  # set your model (specifically the weights(parameters) defined in the model) in GPU

    GAN_Loss = nn.BCELoss()  # define a GAN loss

    G_optim = torch.optim.Adam(G.parameters(), lr=lr, betas=(BETA_1, BETA_2))  # define Adam optimizer for the generator
    D_optim = torch.optim.Adam(D.parameters(), lr=lr, betas=(BETA_1, BETA_2))  # define Adam optimizer for the discriminator

    G_loss_list = []  # an empty list for carrying G_loss
    D_loss_list = []  # an empty list for carrying D_loss
    total_step = 0  # define a variable for counting steps
    for epoch in range(N_EPOCHS):  # for loop with designated epochs
        for i, targets in enumerate(data_loader):  # enumerate the dataset
            total_step += 1  # add the step
            valid = torch.ones([targets.shape[0], 1])  # make a valid grid whose values are all 1
            fake = torch.zeros([targets.shape[0], 1])  # make a fake grid whose vlaues are all 0

            if USE_CUDA:  # if you have GPU device
                valid = valid.cuda()  # set the valid grid in GPU
                fake = fake.cuda()  # set the fake grid in GPU

            reals = targets  # define a variable whose name is reals for the intuition's sake
            z = torch.randn(reals.shape[0], DIM_LATENT)  # make a random normal noise

            if USE_CUDA:  # if you have GPU device
                reals = reals.cuda()  # set the reals in GPU
                z = z.cuda()  # set the noise in GPU

            fakes = G(z)  # generate fake samples

            real_loss = GAN_Loss(D(reals), valid)  # get a loss from reals
            fake_loss = GAN_Loss(D(fakes.detach()), fake)  # get a loss from fakes. detach() ensures that the model
            # won't caluate gradients of parameters involved with generating fakes, i.e. paramteres in the generator.
            # This enables faster learning as the device won't waste the time for calculating unused gradients of
            # parameters in the generator.

            D_loss = (real_loss + fake_loss)*0.5  # total loss for the discriminator
            D_loss_list.append(D_loss.item())

            D_optim.zero_grad()  # zero the gradients of the parameters defined in the discriminator
            D_loss.backward()  # distribute gradients for the parameters defined in the discriminator
            D_optim.step()  # update the parameters with the distributed gradients

            if total_step % N_DISC == 0:  # after N updated of the discriminator
                z = torch.randn(targets.shape[0], DIM_LATENT)  # make a random normal noise
                if USE_CUDA:  # if you have GPU device
                    z = z.cuda()  # set the noise in GPU
                fakes = G(z)  # generate fake samples

                G_loss = GAN_Loss(D(fakes), valid)  # get a generator loss. Note that we didn't detach() this time as
                G_loss_list.append(G_loss.item())
                # we are going to update the generator's parameters.

                G_optim.zero_grad()  # zero the gradients of the parameters defined in the generator
                G_loss.backward()  # distribute gradients for the parameters defined in the generator
                G_optim.step()  # update the parameters with the distributed gradients
                print("Epoch: {}, Total step: {}, D_loss: {:.{prec}}, G_loss: {:.{prec}}"
                      .format(epoch, total_step, D_loss.item(), G_loss.item(), prec=4))
                # print epoch, total step, and the corresponding discriminator loss and generator loss with precision 4

            if total_step % SAVE_INTERVAL == 0:  # for every SAVE_INTERVAL
                save_image(fakes[0].detach(), path=os.path.join('Celeba_' + str(total_step)))  # save image in the path

    print(datetime.datetime.now() - start_time)  # print the total time taken
    plt.figure()  # make a grid
    plt.plot(list(range(len(D_loss_list))), D_loss_list, linestyle='--', label='D loss')  # plot D_loss_list
    plt.plot(list(range(len(G_loss_list))), G_loss_list, linestyle='--', label='G loss')  # plot G_loss_list
    plt.title('Celeba')  # set title as 'MNIST'
    plt.xlabel('Iteration')  # set x axis name as 'Iteration'
    plt.ylabel('GAN loss')  # set y axis name as 'GAN loss'
    plt.legend()  # show labels on the graph
    plt.savefig('GAN_MNIST.png')  # save the graph as the png file
    plt.show()  # show the graph

import torch.nn as nn
import numpy as np
from PIL import Image

def weight_init(module):
    class_name = module.__class__.__name__
    if class_name.find('Conv') != -1:
        module.weight.detach().normal_(0.0, 0.02)

    elif class_name.find('BatchNorm') != -1:
        module.weight.detach().normal_(1.0, 0.02)
        module.bias.detach().fill_(0.0)


def block(module, normalization=True, relu=True, dropout=False):
    layers = []

    if relu:
        layers.append(nn.ReLU(inplace=True))
    elif not relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers.append(module)

    if normalization:
        layers.append(nn.BatchNorm2d(module.weight.size()[0]))

    if dropout:
        layers.append(nn.Dropout2d(0.5, inplace=True))

    return nn.Sequential(*layers)


def get_grid_shape(opt):
    assert opt.patch_size in [1, 16, 70, 286]
    patch_to_grid_dict = {70: (30, 30)}

    return (opt.batch_size, 1, *patch_to_grid_dict[opt.patch_size])


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

\




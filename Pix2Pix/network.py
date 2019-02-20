import torch
from math import log2
from utils import *


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        act_down = nn.LeakyReLU(0.2, inplace=True)
        act_up = nn.ReLU(inplace=True)
        image_height = opt.image_height
        input_ch = opt.input_ch
        max_ch = opt.max_ch
        n_downsample = int(log2(image_height))
        n_gf = opt.n_gf
        norm = nn.BatchNorm2d
        output_ch = opt.output_ch

        idx_max_ch = int(log2(max_ch // n_gf))
        for i in range(n_downsample):
            if i == 0:
                down_block = [nn.Conv2d(input_ch, n_gf, kernel_size=4, padding=1, stride=2, bias=False)]
                up_block = [act_up,
                            nn.ConvTranspose2d(2 * n_gf, output_ch, kernel_size=4, padding=1, stride=2, bias=False),
                            nn.Tanh()]

            elif 1 <= i <= idx_max_ch:
                down_block = [act_down, nn.Conv2d(n_gf, 2 * n_gf, kernel_size=4, padding=1, stride=2, bias=False),
                              norm(2 * n_gf)]
                up_block = [act_up, nn.ConvTranspose2d(4 * n_gf, n_gf, kernel_size=4, padding=1, stride=2, bias=False),
                            norm(n_gf)]

            elif idx_max_ch < i < n_downsample - 4:
                down_block = [act_down, nn.Conv2d(n_gf, n_gf, kernel_size=4, padding=1, stride=2, bias=False),
                              norm(2 * n_gf)]
                up_block = [act_up, nn.ConvTranspose2d(2 * n_gf, n_gf, kernel_size=4, padding=1, stride=2, bias=False),
                            norm(n_gf)]

            elif n_downsample - 4 <= i < n_downsample - 2:
                down_block = [act_down, nn.Conv2d(n_gf, n_gf, kernel_size=4, padding=1, stride=2, bias=False),
                              norm(n_gf)]
                up_block = [act_up, nn.ConvTranspose2d(2 * n_gf, n_gf, kernel_size=4, padding=1, stride=2, bias=False),
                            norm(n_gf), nn.Dropout2d(0.5, inplace=True)]

            else:
                down_block = [act_down, nn.Conv2d(n_gf, n_gf, kernel_size=4, padding=1, stride=2, bias=False)]
                up_block = [act_up, nn.ConvTranspose2d(n_gf, n_gf, kernel_size=4, padding=1, stride=2, bias=False),
                            norm(n_gf), nn.Dropout2d(0.5, inplace=True)]

            self.add_module('Down_block_{}'.format(i), nn.Sequential(*down_block))
            self.add_module('Up_block_{}'.format(i), nn.Sequential(*up_block))
            n_gf *= 2 if n_gf < max_ch else None

        self.n_downsample = n_downsample

    def forward(self, x):
        layers = [x]
        for i in range(self.n_downsample):
            layers += [getattr(self, 'Down_block_{}'.format(i))(layers[-1])]
        x = getattr(self, 'Up_block_{}'.format(self.n_downsample - 1))(layers[-1])
        for i in range(self.n_downsample - 2, -1, -1):
            x = getattr(self, 'Up_block_{}'.format(i))(torch.cat([x, layers[i]], dim=1))
        return x


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        act = nn.LeakyReLU(0.2, inplace=True)
        input_ch = opt.input_ch + opt.output_ch
        n_df = opt.n_df
        norm = nn.BatchNorm2d
        patch_size = opt.patch_size

        if patch_size == 1:
            model = [nn.Conv2d(input_ch, n_df, kernel_size=1, bias=False), act]
            model += [nn.Conv2d(n_df, 2 * n_df, kernel_size=1, bias=False), norm(2 * n_df), act]
            model += [nn.Conv2d(2 * n_df, 1, kernel_size=1, bias=False)]

        elif patch_size == 16:
            model = [nn.Conv2d(input_ch, n_df, kernel_size=4, padding=1, stride=2, bias=False), act]
            model += [nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2, bias=False), norm(2 * n_df), act]
            model += [nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=1, bias=False), norm(4 * n_df),
                      act]
            model += [nn.Conv2d(4 * n_df, 1, kernel_size=4, padding=1, stride=1, bias=False)]

        elif patch_size == 70:
            model = [nn.Conv2d(input_ch, n_df, kernel_size=4, padding=1, stride=2, bias=False), act]
            model += [nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2, bias=False), norm(2 * n_df), act]
            model += [nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2, bias=False), norm(4 * n_df),
                      act]
            model += [nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=1, bias=False), norm(8 * n_df),
                      act]
            model += [nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1, stride=1, bias=False)]

        elif patch_size == 286:
            model = [nn.Conv2d(input_ch, n_df, kernel_size=4, padding=1, stride=2, bias=False), act]
            model += [nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2, bias=False), norm(2 * n_df), act]
            model += [nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2, bias=False), norm(4 * n_df),
                      act]
            model += [nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=2, bias=False), norm(8 * n_df),
                      act]
            model += [nn.Conv2d(8 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=1, bias=False), norm(8 * n_df)]
            model += [nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1, stride=1, bias=False)]

        else:
            raise NotImplementedError("Invalid patch size {}. Please choose among [1, 16, 70, 286].".format(patch_size))

        model += [nn.Sigmoid()]
        self.add_module('Model', nn.Sequential(*model))

    def forward(self, x):
        return getattr(self, 'Model')(x)

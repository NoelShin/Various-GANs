import torch
from utils import *

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt
        self.set_weight()
        self.build_model()

    def set_weight(self):
        input_channel = self.opt.input_channel  # 1
        output_channel = self.opt.output_channel  # 3
        n_gf = self.opt.n_gf  # 64

        # Encoder network starts
        self.first_conv = nn.Conv2d(input_channel, n_gf, kernel_size=4, padding=1, stride=2, bias=False)  # 1x64x128x128
        self.conv_1 = nn.Conv2d(n_gf, 2 * n_gf, kernel_size=4, padding=1, stride=2, bias=False)  # 1x128x64x64
        self.conv_2 = nn.Conv2d(2 * n_gf, 4 * n_gf, kernel_size=4, padding=1, stride=2, bias=False)  # 1x256x32x32
        self.conv_3 = nn.Conv2d(4 * n_gf, 8 * n_gf, kernel_size=4, padding=1, stride=2, bias=False)  # 1x512x16x16
        self.conv_4 = nn.Conv2d(8 * n_gf, 8 * n_gf, kernel_size=4, padding=1, stride=2, bias=False)  # 1x512x8x8
        self.conv_5 = nn.Conv2d(8 * n_gf, 8 * n_gf, kernel_size=4, padding=1, stride=2, bias=False)  # 1x512x4x4
        self.conv_6 = nn.Conv2d(8 * n_gf, 8 * n_gf, kernel_size=4, padding=1, stride=2, bias=False)  # 1x512x2x2
        self.conv_7 = nn.Conv2d(8 * n_gf, 8 * n_gf, kernel_size=4, padding=1, stride=2, bias=False)  # 1x512x1x1

        # Encoder network ends and Decoder network start
        self.dconv_7 = nn.ConvTranspose2d(8 * n_gf, 8 * n_gf, kernel_size=4, padding=1, stride=2, bias=False)  # 1x512x2x2
        self.dconv_6 = nn.ConvTranspose2d(2 * 8 * n_gf, 8 * n_gf, kernel_size=4, padding=1, stride=2, bias=False)  # 1x512x4x4
        self.dconv_5 = nn.ConvTranspose2d(2 * 8 * n_gf, 8 * n_gf, kernel_size=4, padding=1, stride=2, bias=False)  # 1x512x8x8
        self.dconv_4 = nn.ConvTranspose2d(2 * 8 * n_gf, 8 * n_gf, kernel_size=4, padding=1, stride=2, bias=False)  # 1x512x16x16
        self.dconv_3 = nn.ConvTranspose2d(2 * 4 * n_gf, 4 * n_gf, kernel_size=4, padding=1, stride=2, bias=False)  # 1x256x32x32
        self.dconv_2 = nn.ConvTranspose2d(2 * 2 * n_gf, 2 * n_gf, kernel_size=4, padding=1, stride=2, bias=False)  # 1x128x64x64
        self.dconv_1 = nn.ConvTranspose2d(2 * n_gf, n_gf, kernel_size=4, padding=1, stride=2, bias=False)  # 1x64x128x128
        self.last_dconv = nn.ConvTranspose2d(n_gf, output_channel, kernel_size=4, padding=1, stride=2, bias=False)  # 1x3x128x128

    def build_model(self):
        self.first_block = block(self.conv_1, relu=False)
        self.second_block = block(self.conv_2, relu=False)
        self.third_block = block(self.conv_3, relu=False)
        self.fourth_block = block(self.conv_4, relu=False)
        self.fifth_block = block(self.conv_5, relu=False)
        self.sixth_block = block(self.conv_6, relu=False)
        self.seventh_block = block(self.conv_7, relu=False, normalization=False)

        self.seventh_dblock = block(self.dconv_7, dropout=True)
        self.sixth_dblock = block(self.dconv_6, dropout=True)
        self.fifth_dblock = block(self.dconv_5, dropout=True)
        self.fourth_dblock = block(self.dconv_4)
        self.third_dblock = block(self.dconv_3)
        self.second_dblock = block(self.dconv_2)
        self.first_dblock = block(self.dconv_1)

    def forward(self, x):
        d0 = self.first_conv(x)
        d1 = self.first_block(d0)
        d2 = self.second_block(d1)
        d3 = self.third_block(d2)
        d4 = self.fourth_block(d3)
        d5 = self.fifth_block(d4)
        d6 = self.sixth_block(d5)  # 1z512x2x2
        d7 = self.seventh_block(d6)  # 1x512x1x1
        u7 = self.seventh_dblock(d7)  # 1x512x2x2
        u6 = self.sixth_dblock(torch.cat([u7, d6], axis=1))  # 1x512x4x4
        u5 = self.fifth_dblock(torch.cat([u6, d5], axis=1))  # 1x512x8x8
        u4 = self.fourth_dblock(torch.cat([u5, d4], axis=1))  # 1x512x16x16
        u3 = self.third_dblock(torch.cat([u4, d3], axis=1))  # 1x512x32x32
        u2 = self.second_dblock(torch.cat([u3, d2], axis=1))  # 1x512x64x64
        u1 = nn.relu()(self.first_dblock(torch.cat([u2, d1], axis=1)))  # 1x512x128x128
        u0 = nn.Tanh()(self.last_dconv(u1))  # 1x512x256x256

        return u0


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.opt = opt
        self.set_weight()
        self.build_model()

    def set_weight(self):
        input_channel = self.opt.output_channel
        n_df = self.opt.n_df

        self.conv_1 = nn.Conv2d(input_channel, n_df, kernel_size=4, padding=1, stride=2, bias=False)  # 1x64x128x128
        self.conv_2 = nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2, bias=False)  # 1x128x64x64
        self.conv_3 = nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=2, bias=False)  # 1x256x32x32
        self.conv_4 = nn.Conv2d(8 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=1, bias=False)  # 1x512x31x31
        self.conv_5 = nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1, stride=1, bias=False)  # 1x512x30x30

    def build_model(self):
        self.model = nn.Sequential()
        self.model.add_module('first_conv', self.conv_1)

        self.model.add_module('first_block', block(self.conv_2, relu=False))
        self.model.add_module('second_block', block(self.conv_3, relu=False))
        self.model.add_module('third_block', block(self.conv_4, relu=False))

        self.model.add_module('last_leaky', nn.LeakyReLU(0.2, inplace=True))
        self.model.add_module('last_conv', self.conv_5)
        self.model.add_module('sigmoid', nn.Sigmoid())

    def forward(self, x):
        return self.model(x)

# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/9/11 0011, matt '

import sys
sys.path.append("..")

import torch
import torch.nn as nn

import config


class DC_Generator(nn.Module):
    def __init__(self, input_dim, num_filter, output_dim, verbose=False):
        super(DC_Generator, self).__init__()

        self.hidden_layer = nn.Sequential()
        for i in range(len(num_filter)):
            if i == 0:
                deconv = nn.ConvTranspose2d(input_dim, num_filter[i], kernel_size=4, stride=1, padding=0,
                                            output_padding=0, bias=False)
            else:
                deconv = nn.ConvTranspose2d(num_filter[i-1], num_filter[i], kernel_size=5, stride=2, padding=2,
                                            output_padding=1, bias=False)

            deconv_name = 'deconv' + str(i+1)
            self.hidden_layer.add_module(deconv_name, deconv)

            bn_name = "bn" + str(i+1)
            self.hidden_layer.add_module(bn_name, nn.BatchNorm2d(num_filter[i]))

            act_name = "act" + str(i+1)
            self.hidden_layer.add_module(act_name, nn.ReLU())

        self.output_layer = nn.Sequential()

        out = nn.ConvTranspose2d(num_filter[i], output_dim, kernel_size=5, stride=2, padding=2, output_padding=1,
                                 bias=False)
        self.output_layer.add_module('out', out)
        self.output_layer.add_module('act', torch.nn.Tanh())
        self.verbose = verbose

    def forward(self, x):
        if self.verbose:
            print("input x: ", x.shape)
        h = self.hidden_layer(x)
        if self.verbose:
            print("hidden_layer x: ", h.shape)
        out = self.output_layer(h)
        if self.verbose:
            print("output_layer x: ", out.shape)
        return out


class DC_Discriminator(nn.Module):
    def __init__(self, input_dim, num_filters, output_dim, verbose=False):
        super(DC_Discriminator, self).__init__()

        self.hidden_layer = nn.Sequential()
        for i in range(len(num_filters)):
            if i == 0:
                conv = nn.Conv2d(input_dim, num_filters[i], kernel_size=4, stride=2, padding=1)
            else:
                conv = nn.Conv2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1)

            conv_name = 'conv' + str(i+1)
            self.hidden_layer.add_module(conv_name, conv)

            bn_name = 'bn' + str(i+1)
            self.hidden_layer.add_module(bn_name, nn.InstanceNorm2d(num_filters[i], affine=True))

            act_name = "act" + str(i + 1)
            self.hidden_layer.add_module(act_name, nn.LeakyReLU(0.2))

        self.output_layer = nn.Sequential()
        out = nn.Conv2d(num_filters[i], output_dim, kernel_size=4, stride=1, padding=0)
        self.output_layer.add_module("out", out)
        self.verbose = verbose

    def forward(self, x):
        if self.verbose:
            print("input x: ", x.shape)
        h = self.hidden_layer(x)
        if self.verbose:
            print("hidden_layer x: ", h.shape)
        out = self.output_layer(h)
        if self.verbose:
            print("output_layer x: ", out.shape)
        return out.view(out.shape[0])


class Generator(nn.Module):
    def __init__(self, ch=8):
        super(Generator, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(100, ch * 8, 4, 1, 0, 0, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ch * 8, ch * 4, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ch * 4, ch * 2, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ch * 2, ch, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ch, 1, 5, 2, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input_):
        return self.conv(input_)


class Discriminator(nn.Module):
    def __init__(self, ch=8):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch, ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch * 4, ch * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input_):
        return self.conv(input_).view(input_.size(0))


if __name__ == "__main__":
    DC_GNet = DC_Generator(config.G_input_dim, config.num_filters, config.G_output_dim, verbose=True)
    DC_DNet = DC_Discriminator(config.D_input_dim, config.num_filters[::-1], config.D_output_dim, verbose=True)

    z_ = torch.randn(32, config.G_input_dim).view(-1, config.G_input_dim, 1, 1)
    gen_img = DC_GNet(z_)
    print("predict result: ", DC_DNet(gen_img).shape)
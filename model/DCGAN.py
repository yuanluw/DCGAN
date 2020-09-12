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
                deconv = nn.ConvTranspose2d(input_dim, num_filter[i], kernel_size=4, stride=1, padding=0)
            else:
                deconv = nn.ConvTranspose2d(num_filter[i-1], num_filter[i], kernel_size=4, stride=2, padding=1)

            deconv_name = 'deconv' + str(i+1)
            self.hidden_layer.add_module(deconv_name, deconv)
            nn.init.normal_(deconv.weight, mean=0.0, std=0.02)
            nn.init.constant_(deconv.bias, 0.0)

            bn_name = "bn" + str(i+1)
            self.hidden_layer.add_module(bn_name, nn.BatchNorm2d(num_filter[i]))

            act_name = "act" + str(i+1)
            self.hidden_layer.add_module(act_name, nn.ReLU())

        self.output_layer = nn.Sequential()

        out = nn.ConvTranspose2d(num_filter[i], output_dim, kernel_size=4, stride=2, padding=1)
        self.output_layer.add_module('out', out)
        nn.init.normal_(out.weight, mean=0.0, std=0.02)
        nn.init.constant_(out.bias, 0.0)
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

            nn.init.normal_(conv.weight, mean=0.0, std=0.02)
            nn.init.constant_(conv.bias, 0.0)

            if i != 0:
                bn_name = 'bn' + str(i+1)
                self.hidden_layer.add_module(bn_name, nn.BatchNorm2d(num_filters[i]))

            act_name = "act" + str(i + 1)
            self.hidden_layer.add_module(act_name, nn.LeakyReLU(0.2))

        self.output_layer = nn.Sequential()
        out = nn.Conv2d(num_filters[i], output_dim, kernel_size=4, stride=1, padding=0)
        self.output_layer.add_module("out", out)
        nn.init.normal_(out.weight, mean=0.0, std=0.02)
        nn.init.constant_(out.bias, 0.0)
        self.output_layer.add_module('act', nn.Sigmoid())
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


if __name__ == "__main__":
    DC_GNet = DC_Generator(config.G_input_dim, config.num_filters, config.G_output_dim, verbose=True)
    DC_DNet = DC_Discriminator(config.D_input_dim, config.num_filters[::-1], config.D_output_dim, verbose=True)

    z_ = torch.randn(32, config.G_input_dim).view(-1, config.G_input_dim, 1, 1)
    gen_img = DC_GNet(z_)
    print("predict result: ", DC_DNet(gen_img).shape)
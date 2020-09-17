# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/9/11 0011, matt '

import torch

num_worker = 8

image_size = 64

checkpoint_path = '/home/wyl/codeFile/DCGAN/pre_train'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G_input_dim = 100
mnist_G_output_dim = 1
mnist_D_input_dim = 1

cifar_G_output_dim = 3
cifar_D_input_dim = 3

D_output_dim = 1
num_filters = [256, 128, 64, 16]

save_dir = '/home/wyl/codeFile/DCGAN/result'

print_freq = 20

clip = 0.01

n_critic = 5

lambda_gp = 10

nrow = 8

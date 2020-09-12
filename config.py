# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/9/11 0011, matt '

import torch

num_worker = 8

image_size = 32

checkpoint_path = '/home/wyl/codeFile/DCGAN/pre_train'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G_input_dim = 100
G_output_dim = 1
D_input_dim = 1
D_output_dim = 1
num_filters = [1024, 512, 256, 128]

save_dir = '/home/wyl/codeFile/DCGAN/result'

print_freq = 100

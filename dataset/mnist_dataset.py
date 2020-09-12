# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/9/11 0011, matt '

import sys
sys.path.append("..")

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

import config

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


def get_mnist_dataset(state="train", batch_size=64):

    trans = None
    if state == "train":
        trans = train_transform
    else:
        trans = test_transform

    dataset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=trans)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config.num_worker)
    return loader



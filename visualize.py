# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/9/18 0018, matt '

import os
import imageio

import config


def generate_gif(name, dataset_name, net_name):
    gif_images = []
    for i in range(100):
        gif_images.append(imageio.imread(os.path.join(config.save_dir, f"{dataset_name}_{net_name}_{i+1}_result.jpg")))

    imageio.mimsave(name, gif_images, fps=5)


if __name__ == "__main__":
    generate_gif("mnist_DCGAN.gif", "mnist", "DCGAN")
    generate_gif("cifar_DCGAN.gif", "cifar", "DCGAN")



# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/9/11 0011, matt '

import os
import logging

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import autograd
import torchvision
import torchsnooper

import config

cur_path = os.path.abspath(os.path.dirname(__file__))
Tensor = torch.cuda.FloatTensor if config.device else torch.FloatTensor


def de_norm(x):
    out = (x + 1)/2
    return out.clamp(0, 1)


def plot_loss(d_losses, g_losses, num_epoch, net_name="", dataset_name=""):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epoch+1)
    ax.set_ylim(min(np.min(g_losses), np.min(d_losses))*1.1, max(np.max(g_losses), np.max(d_losses))*1.1)
    plt.xlabel(f'Epoch {num_epoch+1}')
    plt.ylabel("loss values")
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.legend()

    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    save_fn = os.path.join(config.save_dir, f"{dataset_name}_{net_name}_{num_epoch+1}_loss.png")
    plt.savefig(save_fn)


def plot_result(generator, noise, num_epoch, fig_size=(5, 5), net_name="", dataset_name=""):
    generator.eval()

    noise = noise.to(config.device)
    gen_img = generator(noise)
    gen_img = de_norm(gen_img)

    generator.train()

    n_rows = np.sqrt(noise.size()[0]).astype(np.int32)
    n_cols = np.sqrt(noise.size()[0]).astype(np.int32)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    for ax, img in zip(axes.flatten(), gen_img):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        ax.imshow(img.cpu().data.view(config.image_size, config.image_size).numpy(), cmap='gray', aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.text(0.5, 0.04, f"epoch{num_epoch+1}", ha="center")

    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    save_fn = os.path.join(config.save_dir, f"{dataset_name}_{net_name}_{num_epoch+1}_result.png")
    plt.savefig(save_fn)


# @torchsnooper.snoop()
def plot_sample(generator, noise, num_epoch, net_name="", dataset_name=""):
    generator.eval()

    os.makedirs(config.save_dir, exist_ok=True)
    f = os.path.join(config.save_dir, f"{dataset_name}_{net_name}_{num_epoch+1}_result.jpg")
    torchvision.utils.save_image(generator(noise).data, f, normalize=True, nrow=config.nrow)

    generator.train()


def count_time(prev_time, cur_time):
    h, reminder = divmod((cur_time-prev_time).seconds, 3600)
    m, s = divmod(reminder, 60)
    time_str = "time %02d:%02d:%02d" %(h, m, s)
    return time_str


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(dataset_name, net_name, epoch, g_net, d_net):

    torch.save(g_net.state_dict(), os.path.join(config.checkpoint_path, dataset_name + "_" + net_name + "_g_net_" +
                                                f"%2.3f" % epoch + '.pth'))
    torch.save(d_net.state_dict(), os.path.join(config.checkpoint_path, dataset_name + "_" + net_name + "_d_net_" +
                                                f"%2.3f" % epoch + '.pth'))


def load_weight(pretrained_dict, model):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 更新当前网络的结构字典
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def calc_gradient_penalty(D, real_samples, fake_samples):

    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    alpha = alpha.to(config.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    real_samples = real_samples.shape[0]
    fake = Tensor(real_samples).fill_(1.0)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
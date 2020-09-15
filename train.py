# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/9/11 0011, matt '


import os
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from dataset.mnist_dataset import get_mnist_dataset
from dataset.cifar_dataset import get_cifar_dataset
from model.DCGAN import DC_Generator, DC_Discriminator, Generator, Discriminator
import config
from utils import count_time, get_logger, AverageMeter, plot_loss, plot_sample, save_checkpoint, load_weight, \
    gradient_penalty

cur_path = os.path.abspath(os.path.dirname(__file__))


def run(arg):
    # torch.manual_seed(7)
    # np.random.seed(7)
    print("lr %f, epoch_num %d, decay_rate %f gamma %f" % (arg.lr, arg.epochs, arg.decay, arg.gamma))

    print("====>Loading data")
    if arg.dataset == 'mnist':
       train_data = get_mnist_dataset("train", arg.batch_size)
    elif arg.dataset == 'cifar':
        train_data = get_cifar_dataset("train", arg.batch_size)

    print("====>Building model")
    if arg.net == "DCGAN":
        g_net = DC_Generator(config.G_input_dim, config.num_filters, config.G_output_dim, verbose=False)
        d_net = DC_Discriminator(config.D_input_dim, config.num_filters[::-1], config.D_output_dim, verbose=False)

        # g_net = Generator()
        # d_net = Discriminator()

        g_net = g_net.to(config.device)
        d_net = d_net.to(config.device)

    g_optimizer = optim.RMSprop(g_net.parameters(), lr=arg.lr)
    d_optimizer = optim.RMSprop(d_net.parameters(), lr=arg.lr)

    if arg.mul_gpu:
        g_net = nn.DataParallel(g_net)
        d_net = nn.DataParallel(d_net)

    if arg.checkpoint is not None:
        print("load pre train model")
        g_pretrained_dict = torch.load(os.path.join(config.checkpoint_path, arg.dataset + "_" +
                                                      arg.net + "_g_net_" + arg.checkpoint + '.pth'))
        d_pretrained_dict = torch.load(os.path.join(config.checkpoint_path, arg.dataset + "_" +
                                                    arg.net + "_d_net_" + arg.checkpoint + '.pth'))
        g_net = load_weight(g_pretrained_dict, g_net)
        d_net = load_weight(d_pretrained_dict, d_net)

    # 日志系统
    logger = get_logger()

    criterion = nn.BCELoss()

    print('Total params: %.2fM' % ((sum(p.numel() for p in g_net.parameters()) +
                                    sum(p.numel() for p in d_net.parameters())) / 1000000.0))
    print("start training: ", datetime.now())
    start_epoch = 0

    fix_noise = torch.autograd.Variable(torch.randn(config.nrow ** 2, config.G_input_dim).view(-1, config.G_input_dim,
                                                                                               1, 1).cuda())
    g_losses = []
    d_losses = []
    for epoch in range(start_epoch, arg.epochs):
        prev_time = datetime.now()

        g_loss, d_loss = train(train_data, g_net, d_net, criterion, g_optimizer, d_optimizer, epoch, logger)
        now_time = datetime.now()
        time_str = count_time(prev_time, now_time)
        print("train: current (%d/%d) batch g_loss is %f d_loss is %f time "
              "is %s" % (epoch, arg.epochs,  g_loss, d_loss, time_str))

        g_losses.append(g_loss)
        d_losses.append(d_loss)

        plot_loss(d_losses, g_losses, epoch, arg.net, arg.dataset)
        plot_sample(g_net, fix_noise, epoch, net_name=arg.net, dataset_name=arg.dataset)

        if epoch % 2 == 0:
            save_checkpoint(arg.dataset, arg.net,  epoch, g_net, d_net)

    save_checkpoint(arg.dataset, arg.net, arg.epochs, g_net, d_net)


def train(train_data, g_net, d_net, criterion, g_optimizer, d_optimizer, epoch, logger):
    g_net.train()
    d_net.train()
    g_losses = AverageMeter()
    d_losses = AverageMeter()

    for i, (img, _) in enumerate(train_data):
        mini_batch = img.size()[0]
        # train discriminator
        x_ = Variable(img.cuda())
        z_ = torch.randn(mini_batch, config.G_input_dim).view(-1, config.G_input_dim, 1, 1)
        z_ = Variable(z_.cuda())

        gen_img = g_net(z_).detach()
        d_loss = -d_net(x_).mean() + d_net(gen_img).mean()

        # bp
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        for p in d_net.parameters():
            p.data.clamp_(-config.clip, config.clip)

        if i % config.n_critic == 0:
            z_ = torch.randn(mini_batch, config.G_input_dim).view(-1, config.G_input_dim, 1, 1)
            z_ = Variable(z_.cuda())
            gen_img = g_net(z_)
            g_loss = -d_net(gen_img).mean()
            # bp
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            g_losses.update(g_loss.item())
            d_losses.update(d_loss.item())

            if i % config.print_freq == 0:
                logger.info('Epoch: [{0}][{1}][{2}]\t'
                            'g_loss {g_loss.val:.4f} ({g_loss.avg:.4f})\t'
                            'd_loss {d_loss.val:.3f} ({d_loss.avg:.3f})'
                            .format(epoch, i, len(train_data), g_loss=g_losses, d_loss=d_losses))

    return g_losses.avg, d_losses.avg
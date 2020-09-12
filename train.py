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
from model.DCGAN import DC_Generator, DC_Discriminator
import config
from utils import count_time, get_logger, AverageMeter, plot_loss, plot_result, save_checkpoint, load_weight

cur_path = os.path.abspath(os.path.dirname(__file__))


def run(arg):
    torch.manual_seed(7)
    np.random.seed(7)
    print("lr %f, epoch_num %d, decay_rate %f gamma %f" % (arg.lr, arg.epochs, arg.decay, arg.gamma))

    print("====>Loading data")
    if arg.dataset == 'mnist':
       train_data = get_mnist_dataset("train", arg.batch_size)

    print("====>Building model")
    if arg.net == "DCGAN":
        g_net = DC_Generator(config.G_input_dim, config.num_filters, config.G_output_dim, verbose=True)
        d_net = DC_Discriminator(config.D_input_dim, config.num_filters[::-1], config.D_output_dim, verbose=True)

        g_net = g_net.to(config.device)
        d_net = d_net.to(config.device)

    g_optimizer = optim.Adam(g_net.parameters(), lr=arg.lr, weight_decay=arg.decay)
    d_optimizer = optim.Adam(d_net.parameters(), lr=arg.lr, weight_decay=arg.decay)

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
    num_test_sample = 5*5
    fix_noise = torch.randn(num_test_sample, config.G_input_dim).view(-1, config.G_input_dim, 1, 1)
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
        plot_result(g_net, fix_noise, epoch, net_name=arg.net, dataset_name=arg.dataset)

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
        x_ = Variable(img.cuda())
        # labels
        y_real = Variable(torch.ones(mini_batch).cuda())
        y_fake = Variable(torch.zeros(mini_batch).cuda())

        # train discriminator with real image
        d_real_output = d_net(x_).squeeze()
        d_real_loss = criterion(d_real_output, y_real)

        # train discriminator with fake image
        z_ = torch.randn(mini_batch, config.G_input_dim).view(-1, config.G_input_dim, 1, 1)
        z_ = Variable(z_.cuda())
        gen_img = g_net(z_)

        d_fake_output = d_net(gen_img).squeeze()
        d_fake_loss = criterion(d_fake_output, y_fake)

        # bp
        d_loss = d_real_loss + d_fake_loss
        d_net.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # train generator
        z_ = torch.randn(mini_batch, config.G_input_dim).view(-1, config.G_input_dim, 1, 1)
        z_ = Variable(z_.cuda())
        gen_img = g_net(z_)

        d_fake_output = d_net(gen_img).squeeze()
        g_loss = criterion(d_fake_output, y_real)

        # bp
        g_net.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        g_losses.update(g_loss.item())
        d_losses.update(d_loss.item())

        if i % config.print_freq == 0:
            logger.info('Epoch: [{0}][{1}][{2}]\t'
                        'g_loss {g_loss.val:.4f} ({g_loss.avg:.4f})\t'
                        'd_loss {d_loss.val:.3f} ({d_loss.avg:.3f})'
                        .format(epoch, i, len(train_data), g_loss=g_loss, d_loss=d_loss))

    return g_losses.avg, d_losses.avg
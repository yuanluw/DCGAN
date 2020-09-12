# -*- coding:utf-8 -*-

__author__ = 'matt'
__email__ = 'mattemail@foxmail.com'
__copyright__ = 'Copyright @ 2020/9/11 0011, matt '


import argparse


def get_augments():
    parser = argparse.ArgumentParser(description="pytorch DCGAN")

    parser.add_argument("--action", type=str, default="train", choices=("train", ))

    parser.add_argument("--net", type=str, default="DCGAN", choices=("DCGAN", ))
    parser.add_argument("--dataset", type=str, default="mnist", choices=("mnist", ))
    parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate(default: 0.0001)")
    parser.add_argument("--decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--gamma", type=float, default=0.1, help="LR is multiplied by gamma on schedule")
    parser.add_argument("--mul_gpu", type=int, default=0, help="use multiple gpu(default: 0")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint")

    return parser.parse_args()


def main():
    arg = get_augments()
    if arg.action == "train":
        from train import run
        run(arg)


if __name__ == "__main__":
    main()
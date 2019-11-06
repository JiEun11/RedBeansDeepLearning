from __future__ import print_function, division, absolute_import

import os
import argparse
import shutil

import torch
import torch.optim as optim


def is_cuda_available():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device


def save_checkpoint(state, is_best, filename='./saves/checkpoint.pth.tar'):
    if not os.path.isdir('./saves'):
        os.makedirs('./saves')
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, './saves/model_best.pth.tar')
    else:
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, './saves/model_best.pth.tar')


class AverageMeter(object):

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
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

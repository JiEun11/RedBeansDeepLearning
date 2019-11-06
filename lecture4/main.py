import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from model_factory import *
from data_loader import *
from module import *
from utils import *

# hyper-parameters.
parser = argparse.ArgumentParser(description='PyTorch Model Trainer')
parser.add_argument('--data-dir', help='path to dataset.')
parser.add_argument('--arch', default='efficientnet', help='model architecture.')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4).')
parser.add_argument('--epochs', default=2000, type=int, help='number of total epochs to run.')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts).')
parser.add_argument('--batch-size', default=256, type=int, help='mini batch size (default: 256).')
parser.add_argument('--learning-rate', '-lr', default=0.001, type=float, help='initial learning rate.')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum.')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4).')
parser.add_argument('--print-freq', default=10, type=int, help='print frequency (default: 10).')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none).')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set.')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

args = parser.parse_args()
best_prec1 = 0


def main():

    global args, best_prec1

    # creates models, to be modular, will be deprecated at future version.
    if args.pretrained:
        print(":: using pre-trained model '{}'.. ::".format(args.arch))
        model = create_model('efficientnet', pretrained=True)
    else:
        print(":: creating model '{}'.. ::".format(args.arch))
        model = create_model('efficientnet')

    # learning setups.
    device = is_cuda_available()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    # (optional) resume from a checkpoint, to be modular, will be deprecated at future version.
    if args.resume:
        if os.path.isfile(args.resume):
            print(":: loading checkpoint '{}'.. ::".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(":: loaded checkpoints '{}' (epoch {}).. ::".format(args.resume, checkpoint['epoch']))
        else:
            print(":: no checkpoint found at '{}'.. ::".format(args.resume))

    cudnn.benchmark = True

    # get data from dataset.
    train_loader, val_loader = get_dataset_cifar(classes=10, batch_size=args.batch_size, num_workers=args.workers)

    # if evaluate.
    if args.evaluate:
        validate(args, val_loader, model, criterion, device)
        return

    # training session.
    for epoch in range(args.start_epoch, args.epochs):

        train(args, train_loader, model, criterion, optimizer, epoch, device)

        prec1 = validate(args, val_loader, model, criterion, device)

        # save model.
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best)


if __name__ == '__main__':
    main()

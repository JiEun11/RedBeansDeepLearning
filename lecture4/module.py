import time
from utils import *


def train(args, train_loader, model, criterion, optimizer, epoch, device):
    print(":: Training Start.. ::")

    # variables for process visualizer
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode.
    model.train()
    model.to(device)

    end = time.time()

    for idx, (inputs, labels) in enumerate(train_loader):

        # measure data loading time.
        data_time.update(time.time() - end)

        inputs, labels = inputs.to(device), labels.to(device)

        # compute output.
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # process visualizer
        if idx % args.print_freq == 0:
            print('Epoch: [{epoch:4d}/{epochs:4d}] Batch: [{idx:3d}/{len:3d}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1: {top1.val:.2f} ({top1.avg:.3f})\t'
                  'Prec@5: {top5.val:.2f} ({top5.avg:.3f})'.format(epoch=epoch + 1, epochs=args.epochs,
                                                                   idx=idx, len=len(train_loader),
                                                                   batch_time=batch_time,
                                                                   data_time=data_time,
                                                                   loss=losses,
                                                                   top1=top1,
                                                                   top5=top5))


def validate(args, val_loader, model, criterion, device):
    print(":: Validation Start.. ::")

    # variables for process visualizer
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for idx, (inputs, labels) in enumerate(val_loader):

        inputs, labels = inputs.to(device), labels.to(device)

        # compute output.
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # process visualizer
        if idx % args.print_freq == 0:
            print('Batch: [{idx}/{len}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss: {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1: {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5: {top5.val:.3f} ({top5.avg:.3f})'.format(idx=idx, len=len(val_loader),
                                                                   batch_time=batch_time, loss=losses,
                                                                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


def accuracy(outputs, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

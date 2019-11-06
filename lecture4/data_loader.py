from __future__ import print_function, division, absolute_import

import torch
import torchvision
import torchvision.transforms as transforms


def data_preprocessing(method=None):

    preprocessing = method

    if preprocessing is 'simple':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        print("The default preprocessing is simple")
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([transforms.ToTensor(), normalize])

    return transform


def get_dataset_cifar(classes=10, dataset_dir='./datasets/cifar', batch_size=128, num_workers=4):

    transform = data_preprocessing('simple')

    if classes == 10:
        trainset = torchvision.datasets.CIFAR10(root=dataset_dir + '/cifar10', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=dataset_dir + '/cifar10', train=False,
                                               download=True, transform=transform)
    elif classes == 100:
        trainset = torchvision.datasets.CIFAR100(root=dataset_dir + '/cifar100', train=True,
                                                 download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root=dataset_dir + '/cifar100', train=False,
                                                download=True, transform=transform)
    else:
        print("Error: Dataset is not specified!")
        trainset = None
        testset = None

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


if __name__ == '__main__':

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    """
    
    """

    print('\n===> Training Start...')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    epochs = 2

    for epoch in range(epochs):
        print('\n===> epoch %d' % epoch)
        running_loss = 0.0

        for idx, data in enumerate(trainloader):
            """
            
            """

            if idx % 100 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, idx + 1, running_loss / 100))
                running_loss = 0.0

    class_correct = list(0. for idx in range(10))
    class_total = list(0. for idx in range(10))

    with torch.no_grad():
        for data_ in testloader:
            """
            
            """
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

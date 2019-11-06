from __future__ import print_function, division, absolute_import

from models import *
import torchvision.models as models


def create_model(name, pretrained=False):

    if is_resnet(name):
        model = models.resnet152(pretrained=pretrained)
    elif is_mobilenet_v2(name):
        model = models.mobilenet_v2(pretrained=pretrained)
    elif is_efficientnet(name):
        model = EfficientNetB0()
    elif is_custom_net(name):
        model = CustomNet()
    else:
        print("Default is alexnet.")
        model = models.alexnet(pretrained=pretrained)

    return model


def is_resnet(name):
    name = name.lower()
    return name.startswith('resnet')


def is_mobilenet_v2(name):
    name = name.lower()
    return name.startswith('mobilenet_v2')


def is_efficientnet(name):
    name = name.lower()
    return name.startswith('efficientnet')


def is_custom_net(name):
    name = name.lower()
    return name.startswith('customnet')

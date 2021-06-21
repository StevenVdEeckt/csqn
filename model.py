import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import logging
from collections import OrderedDict

logger = logging.getLogger('main')

""" 
    ResNet for 32x32 input with 3 channels (by Yerlan Idelbayev)
    - copied from https://github.com/akamaster/pytorch_resnet_cifar10
"""
def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet_, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        if isinstance(num_classes, int):
            self.classifier = nn.Linear(64, num_classes)
        else:
            self.classifier = {}
            for i in range(len(num_classes)):
                self.classifier[i] = nn.Linear(64, num_classes[i])
                self.add_module('linear' + str(i), self.classifier[i])

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, i):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.classifier[i](out)
        return out


"""
    Returns ResNet-32
    :param list or int num_classes: the number of classes - length of num_classes sets the number of output layers
"""
class ResNet(ResNet_):
    def __init__(self, num_classes):
        super(ResNet, self).__init__(block=BasicBlock, num_blocks=[5, 5, 5], num_classes=num_classes)


"""
    LeNet-small network
    :param int or list n_classes: the number of classes - length of num_classes sets the number of output layers
    :param int n_channels: (optional) number of input channels

"""
class LeNet_Small(nn.Module):
    def __init__(self, n_classes, n_channels=3):
        super(LeNet_Small, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.classifier = {}
        if isinstance(n_classes, int):
            self.classifier = nn.Linear(84, n_classes)
        else:
            self.classifier = {}
            for i in range(len(n_classes)):
                self.classifier[i] = nn.Linear(84, n_classes[i])
                self.add_module('fc3' + str(i), self.classifier[i])

    def forward(self, x, i):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.classifier[i](x)
        return x



"""
    LeNet network
    :param int or list n_classes: the number of classes - length of num_classes sets the number of output layers
    :param int n_channels: (optional) number of input channels

"""
class LeNet(nn.Module):
    def __init__(self, n_classes, n_channels=3):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(50 * 5 * 5, 500)
        self.fc2 = nn.Linear(500, 84)
        if isinstance(n_classes, int):
            self.classifier = nn.Linear(84, n_classes)
        else:
            self.classifier = {}
            for i in range(len(n_classes)):
                self.classifier[i] = nn.Linear(84, n_classes[i])
                self.add_module('fc3' + str(i), self.classifier[i])

    def forward(self, x, i=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 50 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.classifier[i](x) if i is not None else self.classifier(x)
        return x


""" 
    Multi-Layer Perceptron (MLP) with 2 hidden layers
    :param int n_classes: the number of classes
    :param int input_dim: dimension of the input, assuming that the input are images (of size input_dim * input_dim)
    :param int n_channels: number of channels of the input images
    :param int n_hidden: number of neurons in the hidden layers
    :param bool bias: (optional) set to False if no biases are desired
"""
class MLP(nn.Module):
    def __init__(self, n_classes, input_dim=28, n_channels=1, n_hidden=256, bias=True):
        super(MLP, self).__init__()
        self.input_dim, self.n_channels = input_dim, n_channels
        self.linear1 = nn.Linear(n_channels * input_dim * input_dim, n_hidden, bias=bias)
        self.linear2 = nn.Linear(n_hidden, n_hidden, bias=bias)
        self.classifier = nn.Linear(n_hidden, n_classes, bias=bias)
        self.dropout = nn.Dropout(0.25)
        self.act = OrderedDict()
        self.kf = {'input': {}, 'pre-activation': {}, 'post-activation': {}}
        self.layers_in_reverse_order = ['classifier.weight', 'linear2.weight', 'linear1.weight']


    def forward(self, x):
        self.act['linear1'] = x
        self.kf['input']['linear1'] = x
        x = self.linear1(x.view(-1, self.n_channels * self.input_dim * self.input_dim))
        self.kf['pre-activation']['linear1.weight'] = x
        x = self.dropout(F.relu(x))
        self.kf['post-activation']['linear1.weight'] = x
        self.kf['input']['linear1.weight'] = x
        self.act['linear2'] = x
        x = self.linear2(x)
        self.kf['pre-activation']['linear2.weight'] = x
        self.kf['post-activation']['linear2.weight'] = x
        x = self.dropout(F.relu(x))
        self.kf['input']['classifier.weight'] = x
        self.act['classifier'] = x
        x = self.classifier(x)
        self.kf['pre-activation']['classifier.weight'] = x
        return x


"""
    Returns the desired model
    :param str model: must be 'resnet', 'lenet' or 'mlp'
    :param int or list n_classes: number of classes or number of classes per output layer
    :param bool return_shared_layers: if True, returns a list of the layers excluding the classification layer
"""
def get_net(model, n_classes, return_shared_layers=False, **kwargs):
    if model == 'resnet':
        if not return_shared_layers:
            return ResNet(num_classes=n_classes)
        else:
            return ResNet(num_classes=n_classes), [n for n, p in ResNet(num_classes=n_classes).state_dict().items() if 'linear' not in n]
    elif model == 'lenet':
        net = LeNet(n_classes=n_classes, n_channels= kwargs.get('n_channels')) if 'n_channels' in kwargs else LeNet(n_classes=n_classes)
        if not return_shared_layers:
            return net
        else:
            return net, [n for n, p in net.state_dict().items() if 'fc3' not in n]
    elif model == 'lenet-small':
        net = LeNet_Small(n_classes=n_classes, n_channels= kwargs.get('n_channels')) if 'n_channels' in kwargs else LeNet_Small(n_classes=n_classes)
        if not return_shared_layers:
            return net
        else:
            return net, [n for n, p in net.state_dict().items() if 'fc3' not in n]
    elif model == 'mlp':
        return MLP(n_classes=n_classes, bias=kwargs.get('bias') if 'bias' in kwargs else True)
    raise Exception("model must be resnet, lenet, mlp or mlp-small but received %s" % model)


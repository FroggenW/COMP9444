# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.linear = nn.Linear(28*28, 10)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x1 = x.view(x.shape[0], -1)
        x2 = self.linear(x1)
        x3 = self.log_softmax(x2)
        return x3

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.linear1 = nn.Linear(28*28, 90)
        self.linear2 = nn.Linear(90, 10)
        self.tanh = nn.Tanh()
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x1 = x.view(x.shape[0], -1)
        x2 = self.linear1(x1)
        x3 = self.tanh(x2)
        x4 = self.linear2(x3)
        x5 = self.log_softmax(x4)
        return x5

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv2d1 = nn.Conv2d(1, 25, 5, padding = 2)
        self.conv2d2 = nn.Conv2d(25, 50, 5, padding = 2)
        self.linear1 = nn.Linear(39200, 1000)
        self.linear2 = nn.Linear(1000, 10)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax()
        #self.pooling = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.conv2d1(x)
        x2 = self.relu(x1)
        x3 = self.conv2d2(x2)
        x4 = self.relu(x3)
        #x_pooling = self.pooling(x4)
        x5 = x4.view(x4.shape[0], -1)
        x6 = self.linear1(x5)
        #x7 = self.relu(x6)
        x8 = self.linear2(x6)
        x9 = self.log_softmax(x8)
        return x9



# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        self.linear1 = nn.Linear(2, num_hid)
        self.linear2 = nn.Linear(num_hid, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        #output = 0*input[:,0]# CHANGE CODE HERE
        r = torch.norm(input, 2, dim = -1).unsqueeze(-1)
        a = torch.atan2(input[:,1], input[:,0]).unsqueeze(-1)
        x = torch.cat((r,a), -1)
        x1 = x.view(x.shape[0], -1)
        x2 = self.linear1(x1)
        self.hid1 = self.tanh(x2)
        x4 = self.linear2(self.hid1)
        output = self.sigmoid(x4)
        return output

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.linear1 = nn.Linear(2, num_hid)
        self.linear2 = nn.Linear(num_hid, num_hid)
        self.linear3 = nn.Linear(num_hid, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        #output = 0*input[:,0] # CHANGE CODE HERE
        x1 = input.view(input.shape[0], -1)
        x2 = self.linear1(x1)
        self.hid1 = self.tanh(x2)
        x4 = self.linear2(self.hid1)
        self.hid2 = self.tanh(x4)
        x6 = self.linear3(self.hid2)
        output = self.sigmoid(x6)
        return output

class ShortNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(ShortNet, self).__init__()
        self.linear1_2 = nn.Linear(2, num_hid)
        self.linear1_3 = nn.Linear(2, num_hid)
        self.linear1_4 = nn.Linear(2, 1)
        self.linear2_3 = nn.Linear(num_hid, num_hid)
        self.linear2_4 = nn.Linear(num_hid, 1)
        self.linear3_4 = nn.Linear(num_hid, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        #self.relu = nn.ReLU()

    def forward(self, input):
        #output = 0*input[:,0] # CHANGE CODE HERE
        Xinput_hid1 = self.linear1_2(input)
        Xinput_hid2 = self.linear1_3(input)
        Xinput_output = self.linear1_4(input)
        self.hid1 = self.tanh(Xinput_hid1)
        #self.hid1 = self.relu(Xinput_hid1)
        Xhid1_hid2 = self.linear2_3(self.hid1)
        Xhid1_output = self.linear2_4(self.hid1)
        self.hid2 = self.tanh(Xinput_hid2 + Xhid1_hid2)
        #self.hid2 = self.relu(Xinput_hid2 + Xhid1_hid2)
        Xhid2_output = self.linear3_4(self.hid2)
        output = self.sigmoid(Xinput_output + Xhid1_output + Xhid2_output)
        return output

def graph_hidden(net, layer, node):
    plt.clf()
    xrange = torch.arange(start=-7, end=7.1, step=0.01, dtype=torch.float32)
    yrange = torch.arange(start=-6.6, end=6.7, step=0.01, dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1), ycoord.unsqueeze(1)), 1)

    with torch.no_grad():
        net.eval()
        net(grid)
        if layer == 1:
            pred = (net.hid1[:, node] >= 0).float()
        elif layer == 2:
            pred = (net.hid2[:, node] >= 0).float()
        plt.clf()
        plt.pcolormesh(xrange, yrange, pred.cpu().view(yrange.size()[0], xrange.size()[0]), cmap='Wistia')
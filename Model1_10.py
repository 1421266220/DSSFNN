#!/usr/bin/env python
# coding=utf-8
import torch
from torch import nn

from vit_modelForModel1_4 import VisionTransformer
from vit_modelForModel1_4_1 import VisionTransformer as VisionTransformer2

class base(nn.Module):
    def __init__(self, in_channel, out_channel, stand_channel, stride=1, ):
        super(base, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=stand_channel,
                               kernel_size=1, stride=stride, bias=False)

        self.conv2 = nn.Conv1d(in_channels=stand_channel, out_channels=stand_channel,
                               kernel_size=5, stride=1, bias=False, padding=2)

        self.conv3 = nn.Conv1d(in_channels=stand_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)



    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2(out)

        out = self.conv3(out)




        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.base1 = base(in_channel, out_channel, 16)
        self.base2 = base(in_channel, out_channel, 16, )
        self.base3 = base(in_channel, out_channel, 16, )
        self.base4 = base(in_channel, out_channel, 16, )
        # self.base5 = base(in_channel, out_channel,16, )
        # self.base6 = base(in_channel, out_channel,16, )
        # self.base7 = base(in_channel, out_channel,16, )
        # self.base8 = base(in_channel, out_channel, 16, )

    def forward(self, x):
        x1 = self.base1(x)
        x2 = self.base2(x)
        x3 = self.base3(x)
        x4 = self.base4(x)
        # x5 = self.base5(x)
        # x6 = self.base6(x)
        # x7 = self.base7(x)
        # x8 = self.base8(x)


        return x1+x2+x3+x4

class RESstack(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(RESstack, self).__init__()
        self.relu = nn.GELU()
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=stride, bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.block1 = BasicBlock(in_channel=out_channel, out_channel=out_channel)
        self.block2 = BasicBlock(in_channel=out_channel, out_channel=out_channel)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.bn2 = nn.BatchNorm1d(out_channel)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.block1(x)+x
        x = self.relu(x)
        x = self.bn1(x)

        x = self.block2(x)+x
        x = self.relu(x)
        x = self.bn2(x)

        x = self.maxpool(x)
        return x




class RLAttention(nn.Module):
    def __init__(self, num_classes=11):
        super(RLAttention, self).__init__()

        self.layer11 = RESstack(in_channel=2, out_channel=64)
        self.layer12 = RESstack(in_channel=64, out_channel=64)
        self.layer13 = RESstack(in_channel=64, out_channel=64)


        self.transformer1 = VisionTransformer(size=(64, 64),
                                              embed_dim=64,
                                              depth=3,
                                              num_heads=16,
                                              num_classes=32)


        self.fc = nn.Linear(2, 64)
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.fc3 = nn.Linear(1, 64)
        self.fc4 = nn.Linear(128,32)


        self.LSTM1 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.LSTM2 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)

        self.softmax = nn.Softmax()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.GELU()

        self.h1 = torch.zeros(1, 400, 128).to("cuda:0")
        self.c1 = torch.zeros(1, 400, 128).to("cuda:0")
        self.h2 = torch.zeros(1, 400, 128).to("cuda:0")
        self.c2 = torch.zeros(1, 400, 128).to("cuda:0")
    def forward(self, x):


        x1 = x[:, 0:2, :]
        x2 = x[:, 2:4, :]

        x2 = self.layer11(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        x2 = self.layer12(x2)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.layer13(x2)
        x2 = self.bn3(x2)
        x2 = self.relu(x2)

        x2 = self.fc1(x2)

        x_1 = self.transformer1(x2)


        x1 = torch.transpose(x1, 1, 2)
        x1 = self.fc(x1)
        # x_2 = self.transformer2(x1)
        x1, (h_n, c_n) = self.LSTM1(x1, (self.h1, self.c1))
        x1 = x1[:, -1, :]
        x1 = torch.unsqueeze(x1,2)
        x1 = self.fc3(x1)
        x1, (h_n, c_n) = self.LSTM2(x1, (self.h2, self.c2))
        x1 = x1[:, -1, :]

        x_2 = self.fc4(x1)

        x = torch.cat((x_1,x_2),1)
        x = self.fc2(x)

        return x

def model(num_classes=11):
    return RLAttention(num_classes=num_classes)

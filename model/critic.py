import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
from model.trelu import TReLU
from model.utils import *

ACTION_EXPANSION = 45  # action 차원 확장 계수

class SoftQNetwork(nn.Module):
    def __init__(self, in_channels, num_actions, hidden_dim, width, height, model_based):
        super(SoftQNetwork, self).__init__()
        
        self.in_planes = 64
        block, num_blocks = cfg(depth=18)

        self.model_based = model_based
        if not model_based:
            in_channels += num_actions

        # ResNet encoder (no pretrained)
        if width >= 128:
            self.encoder = nn.Sequential(
                conv3x3(in_channels, 64, 2),
                TReLU(),
                self._make_layer(block, 64, num_blocks[0], stride=2),
                self._make_layer(block, 128, num_blocks[1], stride=2),
                self._make_layer(block, 256, num_blocks[2], stride=2),
                self._make_layer(block, 512, num_blocks[3], stride=2),
                nn.AvgPool2d(4)
            )
            conv_last_dim = 512
        elif width >= 64:
            self.encoder = nn.Sequential(
                conv3x3(in_channels, 64, 2),
                TReLU(),
                self._make_layer(block, 64, num_blocks[0], stride=2),
                self._make_layer(block, 128, num_blocks[1], stride=2),
                self._make_layer(block, 256, num_blocks[2], stride=2),
                nn.AvgPool2d(4)
            )
            conv_last_dim = 256
        elif width >= 32:
            self.encoder = nn.Sequential(
                conv3x3(in_channels, 64, 2),
                TReLU(),
                self._make_layer(block, 64, num_blocks[0], stride=2),
                self._make_layer(block, 128, num_blocks[1], stride=2),
                nn.AvgPool2d(4)
            )
            conv_last_dim = 128
        elif width >= 16:
            self.encoder = nn.Sequential(
                conv3x3(in_channels, 64, 2),
                TReLU(),
                self._make_layer(block, 64, num_blocks[0], stride=2),
                self._make_layer(block, 128, num_blocks[1], stride=2),
                nn.AvgPool2d(2)
            )
            conv_last_dim = 128
        elif width >= 8:
            self.encoder = nn.Sequential(
                conv3x3(in_channels, 64, 2),
                TReLU(),
                self._make_layer(block, 64, num_blocks[0], stride=2),
                nn.AvgPool2d(2)
            )
            conv_last_dim = 64
        else:
            print(f'Network undefined for the given size H: {height} W: {width}')
        self.q1_head = nn.Sequential(
            nn.Linear(conv_last_dim * block.expansion, hidden_dim),
            TReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2_head = nn.Sequential(
            nn.Linear(conv_last_dim * block.expansion, hidden_dim),
            TReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.trelu = TReLU()

        # For action channels
        self.width = width
        self.height = height

        self.apply(weights_init_)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def encode(self, obs, action):
        if not self.model_based:
            action_channels = action.view(action.size(0), action.size(1), 1, 1).repeat(1, 1, self.width, self.height)
            obs = torch.cat([obs, action_channels], dim=1)
        x = self.encoder(obs)
        x = x.view(x.size(0), -1)
        return x
        
    def forward(self, obs, action):
        # Encode
        x = self.encode(obs, action)

        # infer Q
        q1 = self.q1_head(x)
        q2 = self.q2_head(x)
        
        return q1, q2


def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (BasicBlock, [2,2,2,2]),
        '34': (BasicBlock, [3,4,6,3]),
        '50': (Bottleneck, [3,4,6,3]),
        '101':(Bottleneck, [3,4,23,3]),
        '152':(Bottleneck, [3,8,36,3]),
    }

    return cf_dict[str(depth)]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                weightNorm(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True)),
            )
        self.relu_1 = TReLU()
        self.relu_2 = TReLU()

    def forward(self, x):
        out = self.relu_1(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu_2(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = weightNorm(nn.Conv2d(in_planes, planes, kernel_size=1, bias=True))
        self.conv2 = weightNorm(nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True))
        self.conv3 = weightNorm(nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=True))
        self.relu_1 = TReLU()
        self.relu_2 = TReLU()
        self.relu_3 = TReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                weightNorm(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True)),
            )

    def forward(self, x):
        out = self.relu_1(self.conv1(x))
        out = self.relu_2(self.conv2(out))
        out = self.conv3(out)
        out += self.shortcut(x)
        out = self.relu_3(out)

        return out


class QNetwork(nn.Module):
    def __init__(self, in_channels, num_actions, hidden_dim, width, height, model_based):
        super(QNetwork, self).__init__()
        
        self.in_planes = 64
        block, num_blocks = cfg(depth=18)

        self.model_based = model_based
        if not model_based:
            in_channels += num_actions

        # ResNet encoder (no pretrained)
        if width >= 128:
            self.encoder = nn.Sequential(
                conv3x3(in_channels, 64, 2),
                TReLU(),
                self._make_layer(block, 64, num_blocks[0], stride=2),
                self._make_layer(block, 128, num_blocks[1], stride=2),
                self._make_layer(block, 256, num_blocks[2], stride=2),
                self._make_layer(block, 512, num_blocks[3], stride=2),
                nn.AvgPool2d(4)
            )
            conv_last_dim = 512
        elif width >= 64:
            self.encoder = nn.Sequential(
                conv3x3(in_channels, 64, 2),
                TReLU(),
                self._make_layer(block, 64, num_blocks[0], stride=2),
                self._make_layer(block, 128, num_blocks[1], stride=2),
                self._make_layer(block, 256, num_blocks[2], stride=2),
                nn.AvgPool2d(4)
            )
            conv_last_dim = 256
        elif width >= 32:
            self.encoder = nn.Sequential(
                conv3x3(in_channels, 64, 2),
                TReLU(),
                self._make_layer(block, 64, num_blocks[0], stride=2),
                self._make_layer(block, 128, num_blocks[1], stride=2),
                nn.AvgPool2d(4)
            )
            conv_last_dim = 128
        elif width >= 16:
            self.encoder = nn.Sequential(
                conv3x3(in_channels, 64, 2),
                TReLU(),
                self._make_layer(block, 64, num_blocks[0], stride=2),
                self._make_layer(block, 128, num_blocks[1], stride=2),
                nn.AvgPool2d(2)
            )
            conv_last_dim = 128
        elif width >= 8:
            self.encoder = nn.Sequential(
                conv3x3(in_channels, 64, 2),
                TReLU(),
                self._make_layer(block, 64, num_blocks[0], stride=2),
                nn.AvgPool2d(2)
            )
            conv_last_dim = 64
        else:
            print(f'Network undefined for the given size H: {height} W: {width}')
        self.q_head = nn.Sequential(
            nn.Linear(conv_last_dim * block.expansion, hidden_dim),
            TReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.trelu = TReLU()

        # For action channels
        self.width = width
        self.height = height

        self.apply(weights_init_)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def encode(self, obs, action):
        if not self.model_based:
            action_channels = action.view(action.size(0), action.size(1), 1, 1).repeat(1, 1, self.width, self.height)
            obs = torch.cat([obs, action_channels], dim=1)
        x = self.encoder(obs)
        x = x.view(x.size(0), -1)
        return x
        
    def forward(self, obs, action):
        # Encode
        x = self.encode(obs, action)

        # infer Q
        q = self.q_head(x)
        
        return q
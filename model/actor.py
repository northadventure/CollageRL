import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from model.trelu import TReLU
from model.utils import *

# SAC policy
class GaussianPolicy(nn.Module):
    def __init__(self, in_channels, num_actions, hidden_dim, action_space=None, width=224, height=224):
        super(GaussianPolicy, self).__init__()
        
        self.in_planes = 64
        block, num_blocks = cfg_bn(depth=18)

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
        self.head = nn.Sequential(
            nn.Linear(conv_last_dim * block.expansion, hidden_dim),
            TReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.trelu = TReLU()

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)
            
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def encode(self, obs):
        x = self.encoder(obs)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, obs):
        x = self.encode(obs)
        
        x = self.trelu(self.head(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


def cfg_bn(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (BasicBlockBN, [2,2,2,2]),
        '34': (BasicBlockBN, [3,4,6,3]),
        '50': (BottleneckBN, [3,4,6,3]),
        '101':(BottleneckBN, [3,4,23,3]),
        '152':(BottleneckBN, [3,8,36,3]),
    }

    return cf_dict[str(depth)]

class BasicBlockBN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockBN, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                (nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)

        return out

class BottleneckBN(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckBN, self).__init__()
        self.conv1 = (nn.Conv2d(in_planes, planes, kernel_size=1, bias=False))
        self.conv2 = (nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        self.conv3 = (nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False))
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                (nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)

        return out



# DDPG policy
class DeterministicPolicy(nn.Module):
    def __init__(self, in_channels, num_actions, hidden_dim, action_space=None, width=224, height=224):
        super(DeterministicPolicy, self).__init__()
        
        self.in_planes = 64
        block, num_blocks = cfg_bn(depth=18)

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
        self.head = nn.Sequential(
            nn.Linear(conv_last_dim * block.expansion, hidden_dim),
            TReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            TReLU(),
            nn.Linear(hidden_dim, num_actions),
            nn.Tanh()
        )

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)
            
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def encode(self, obs):
        x = self.encoder(obs)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, obs):
        # Encode
        x = self.encode(obs)
        action = self.head(x) * self.action_scale + self.action_bias
        return action

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(DeterministicPolicy, self).to(device)
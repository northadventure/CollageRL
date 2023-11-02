import sys
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from model.gradpenalty import gradient_penalty
from model.spnorm import SpectralNorm  # SN-GAN
from model.gradnorm import normalize_gradient  # GN-GAN
import torch.nn.utils.weight_norm as weightNorm
from model.utils import *

class TReLU(nn.Module):
    def __init__(self):
        super(TReLU, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0)

    def forward(self, x):
        x = F.relu(x - self.alpha) + self.alpha
        return x
    

class PureDiscriminator(nn.Module):
    def __init__(self, width=224, height=224):
        super(PureDiscriminator, self).__init__()
        # 8x8
        if width == 8:
            self.encoder = nn.Sequential(
                nn.Conv2d(6, 16, 5, 2, 2),
                TReLU(),
                nn.Conv2d(16, 1, 5, 2, 2),
                nn.AvgPool2d(2)
            )
        # 16x16
        elif width == 16:
            self.encoder = nn.Sequential(
                nn.Conv2d(6, 16, 5, 2, 2),
                TReLU(),
                nn.Conv2d(16, 32, 5, 2, 2),
                TReLU(),
                nn.Conv2d(32, 1, 5, 2, 2),
                nn.AvgPool2d(2)
            )
        # 32x32
        elif width == 32:
            self.encoder = nn.Sequential(
                nn.Conv2d(6, 16, 5, 2, 2),
                TReLU(),
                nn.Conv2d(16, 32, 5, 2, 2),
                TReLU(),
                nn.Conv2d(32, 64, 5, 2, 2),
                TReLU(),
                nn.Conv2d(64, 1, 5, 2, 2),
                nn.AvgPool2d(2)
            )
        # 64x64
        elif width == 64:
            self.encoder = nn.Sequential(
                nn.Conv2d(6, 16, 5, 2, 2),
                TReLU(),
                nn.Conv2d(16, 32, 5, 2, 2),
                TReLU(),
                nn.Conv2d(32, 64, 5, 2, 2),
                TReLU(),
                nn.Conv2d(64, 1, 5, 2, 2),
                nn.AvgPool2d(4)
            )
        # 128x128
        elif width == 128:
            self.encoder = nn.Sequential(
                nn.Conv2d(6, 16, 5, 2, 2),
                TReLU(),
                nn.Conv2d(16, 32, 5, 2, 2),
                TReLU(),
                nn.Conv2d(32, 64, 5, 2, 2),
                TReLU(),
                nn.Conv2d(64, 128, 5, 2, 2),
                TReLU(),
                nn.Conv2d(128, 1, 5, 2, 2),
                nn.AvgPool2d(4)
            )
        elif width == 224:
            # 224x224
            self.encoder = nn.Sequential(
                nn.Conv2d(6, 16, 5, 2, 2),
                TReLU(),
                nn.Conv2d(16, 32, 5, 2, 2),
                TReLU(),
                nn.Conv2d(32, 64, 5, 2, 2),
                TReLU(),
                nn.Conv2d(64, 128, 5, 2, 2),
                TReLU(),
                nn.Conv2d(128, 224, 5, 2, 2),
                TReLU(),
                nn.Conv2d(224, 1, 5, 2, 2),
                nn.AvgPool2d(4)
            )
        else:
            print(f'Undefined discriminator input size (W: {width}, H: {height})')
            sys.exit(0)
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 1)
        return x
    
    def get_sim(self, x):
        return self(x)
    
    
class WeightNormDiscriminator(nn.Module):
    def __init__(self, width=224, height=224):
        super(WeightNormDiscriminator, self).__init__()
        # 8x8
        if width == 8:
            self.encoder = nn.Sequential(
                weightNorm(nn.Conv2d(6, 16, 5, 2, 2)),
                TReLU(),
                weightNorm(nn.Conv2d(16, 1, 5, 2, 2)),
                nn.AvgPool2d(2)
            )
        # 16x16
        elif width == 16:
            self.encoder = nn.Sequential(
                weightNorm(nn.Conv2d(6, 16, 5, 2, 2)),
                TReLU(),
                weightNorm(nn.Conv2d(16, 32, 5, 2, 2)),
                TReLU(),
                weightNorm(nn.Conv2d(32, 1, 5, 2, 2)),
                nn.AvgPool2d(2)
            )
        # 32x32
        elif width == 32:
            self.encoder = nn.Sequential(
                weightNorm(nn.Conv2d(6, 16, 5, 2, 2)),
                TReLU(),
                weightNorm(nn.Conv2d(16, 32, 5, 2, 2)),
                TReLU(),
                weightNorm(nn.Conv2d(32, 64, 5, 2, 2)),
                TReLU(),
                weightNorm(nn.Conv2d(64, 1, 5, 2, 2)),
                nn.AvgPool2d(2)
            )
        # 64x64
        elif width == 64:
            self.encoder = nn.Sequential(
                weightNorm(nn.Conv2d(6, 16, 5, 2, 2)),
                TReLU(),
                weightNorm(nn.Conv2d(16, 32, 5, 2, 2)),
                TReLU(),
                weightNorm(nn.Conv2d(32, 64, 5, 2, 2)),
                TReLU(),
                weightNorm(nn.Conv2d(64, 1, 5, 2, 2)),
                nn.AvgPool2d(4)
            )
        # 128x128
        elif width == 128:
            self.encoder = nn.Sequential(
                weightNorm(nn.Conv2d(6, 16, 5, 2, 2)),
                TReLU(),
                weightNorm(nn.Conv2d(16, 32, 5, 2, 2)),
                TReLU(),
                weightNorm(nn.Conv2d(32, 64, 5, 2, 2)),
                TReLU(),
                weightNorm(nn.Conv2d(64, 128, 5, 2, 2)),
                TReLU(),
                weightNorm(nn.Conv2d(128, 1, 5, 2, 2)),
                nn.AvgPool2d(4)
            )
        elif width == 224:
            # 224x224
            self.encoder = nn.Sequential(
                weightNorm(nn.Conv2d(6, 16, 5, 2, 2)),
                TReLU(),
                weightNorm(nn.Conv2d(16, 32, 5, 2, 2)),
                TReLU(),
                weightNorm(nn.Conv2d(32, 64, 5, 2, 2)),
                TReLU(),
                weightNorm(nn.Conv2d(64, 128, 5, 2, 2)),
                TReLU(),
                weightNorm(nn.Conv2d(128, 224, 5, 2, 2)),
                TReLU(),
                weightNorm(nn.Conv2d(224, 1, 5, 2, 2)),
                nn.AvgPool2d(4)
            )
        else:
            print(f'Undefined discriminator input size (W: {width}, H: {height})')
            sys.exit(0)
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 1)
        return x
    
    def get_sim(self, x):
        return self(x)
    
    
class SpectralNormDiscriminator(nn.Module):
    def __init__(self, width=224, height=224, trelu=False):
        super(SpectralNormDiscriminator, self).__init__()

        if trelu:
            activation = TReLU()
        else:
            activation = nn.ReLU()

        # 8x8
        if width == 8:
            self.encoder = nn.Sequential(
                SpectralNorm(nn.Conv2d(6, 16, 5, 2, 2)),
                activation,
                SpectralNorm(nn.Conv2d(16, 1, 5, 2, 2)),
                nn.AvgPool2d(2)
            )
        # 16x16
        elif width == 16:
            self.encoder = nn.Sequential(
                SpectralNorm(nn.Conv2d(6, 16, 5, 2, 2)),
                activation,
                SpectralNorm(nn.Conv2d(16, 32, 5, 2, 2)),
                activation,
                SpectralNorm(nn.Conv2d(32, 1, 5, 2, 2)),
                nn.AvgPool2d(2)
            )
        # 32x32
        elif width == 32:
            self.encoder = nn.Sequential(
                SpectralNorm(nn.Conv2d(6, 16, 5, 2, 2)),
                activation,
                SpectralNorm(nn.Conv2d(16, 32, 5, 2, 2)),
                activation,
                SpectralNorm(nn.Conv2d(32, 64, 5, 2, 2)),
                activation,
                SpectralNorm(nn.Conv2d(64, 1, 5, 2, 2)),
                nn.AvgPool2d(2)
            )
        # 64x64
        elif width == 64:
            self.encoder = nn.Sequential(
                SpectralNorm(nn.Conv2d(6, 16, 5, 2, 2)),
                activation,
                SpectralNorm(nn.Conv2d(16, 32, 5, 2, 2)),
                activation,
                SpectralNorm(nn.Conv2d(32, 64, 5, 2, 2)),
                activation,
                SpectralNorm(nn.Conv2d(64, 1, 5, 2, 2)),
                nn.AvgPool2d(4)
            )
        # 128x128
        elif width == 128:
            self.encoder = nn.Sequential(
                SpectralNorm(nn.Conv2d(6, 16, 5, 2, 2)),
                activation,
                SpectralNorm(nn.Conv2d(16, 32, 5, 2, 2)),
                activation,
                SpectralNorm(nn.Conv2d(32, 64, 5, 2, 2)),
                activation,
                SpectralNorm(nn.Conv2d(64, 128, 5, 2, 2)),
                activation,
                SpectralNorm(nn.Conv2d(128, 1, 5, 2, 2)),
                nn.AvgPool2d(4)
            )
        elif width == 224:
            # 224x224
            self.encoder = nn.Sequential(
                SpectralNorm(nn.Conv2d(6, 16, 5, 2, 2)),
                activation,
                SpectralNorm(nn.Conv2d(16, 32, 5, 2, 2)),
                activation,
                SpectralNorm(nn.Conv2d(32, 64, 5, 2, 2)),
                activation,
                SpectralNorm(nn.Conv2d(64, 128, 5, 2, 2)),
                activation,
                SpectralNorm(nn.Conv2d(128, 224, 5, 2, 2)),
                activation,
                SpectralNorm(nn.Conv2d(224, 1, 5, 2, 2)),
                nn.AvgPool2d(4)
            )
        else:
            print(f'Undefined discriminator input size (W: {width}, H: {height})')
            sys.exit(0)
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 1)
        return x
    
    def get_sim(self, x):
        return self(x)
    
    
class GAN():
    def __init__(self, type, width, height, device):
        self.type = type
        self.width = width
        self.height = height
        self.device = device
        
        if type in ['wgan']:
            self.netD = WeightNormDiscriminator(width=width, height=height).to(device)
            self.target_netD = WeightNormDiscriminator(width=width, height=height).to(device)
        elif type in ['gngan']:
            self.netD = PureDiscriminator(width=width, height=height).to(device)
            self.target_netD = PureDiscriminator(width=width, height=height).to(device)
        elif type in ['sngan']:
            self.netD = SpectralNormDiscriminator(width=width, height=height, trelu=True).to(device)
            self.target_netD = SpectralNormDiscriminator(width=width, height=height, trelu=True).to(device)

        self.optimizerD = Adam(self.netD.parameters(), lr=3e-4, betas=(0.5, 0.999))

        hard_update(self.target_netD, self.netD)

    def update(self, env, memory, batch_size):
        canvas_batch, source_batch, goal_batch, step_batch, action_batch, shape_batch, reward_batch, next_canvas_batch, next_source_batch, mask_batch = memory.sample(batch_size=batch_size)

        canvas_batch = torch.FloatTensor(canvas_batch).to(self.device)
        goal_batch = torch.FloatTensor(goal_batch).to(self.device)

        fake_data = canvas_batch.detach()
        real_data = goal_batch.detach()

        fake = torch.cat([fake_data, real_data], 1)
        real = torch.cat([real_data, real_data], 1)

        if self.type in ['wgan', 'sngan']:
            D_real = self.netD(real)
            D_fake = self.netD(fake)
        elif self.type in ['gngan']:
            pred = normalize_gradient(self.netD, torch.cat([fake, real], dim=0))
            D_fake, D_real = torch.split(pred, [fake.shape[0], real.shape[0]])

        self.optimizerD.zero_grad()

        if self.type in ['wgan']:
            D_cost = D_fake.mean() - D_real.mean()
            gp = gradient_penalty(self.netD, real, fake, gp_weight=10, device=self.device)
            D_cost = D_cost + gp
        elif self.type in ['sngan', 'gngan']:
            # Original style
            D_cost = nn.BCEWithLogitsLoss()(D_real, torch.ones(fake_data.shape[0], 1).to(self.device)) + \
                     nn.BCEWithLogitsLoss()(D_fake, torch.zeros(fake_data.shape[0], 1).to(self.device))
            # Relativistic style
            # D_cost = nn.BCEWithLogitsLoss()(D_real - D_fake, torch.ones(fake_data.shape[0], 1).to(self.device))
            # Least-square style
            # D_cost = 0.5*(D_real**2).mean() + 0.5*(D_fake**2).mean()

        D_cost.backward()
        self.optimizerD.step()
        soft_update(self.target_netD, self.netD, 0.001)
        # hard_update(self.target_netD, self.netD)

        return D_fake.mean(), D_real.mean(), D_cost.item()

    def similarity(self, fake_data, real_data, grad=False):
        if grad:
            return self.target_netD.get_sim(torch.cat([fake_data, real_data], 1))
        else:
            with torch.no_grad():
                return self.target_netD.get_sim(torch.cat([fake_data, real_data], 1))
        
    def similarity_tensor(self, data, grad=False):
        if grad:
            return self.target_netD.get_sim(data)
        else:
            with torch.no_grad():
                return self.target_netD.get_sim(data)
        

    def save(self, path):
        torch.save(self.netD.state_dict(),'{}/dis.pkl'.format(path))

    def load(self, path):
        self.netD.load_state_dict(torch.load('{}/dis.pkl'.format(path), map_location=self.device))
        hard_update(self.target_netD, self.netD)

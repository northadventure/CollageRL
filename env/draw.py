import torch
import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F

class Shaper64(nn.Module):
    def __init__(self):
        super(Shaper64, self).__init__()
        self.fc1 = (nn.Linear(8, 512))
        self.fc2 = (nn.Linear(512, 1024))
        self.fc3 = (nn.Linear(1024, 2048))
        self.conv1 = (nn.Conv2d(8, 16, 3, 1, 1))
        self.conv2 = (nn.Conv2d(16, 16, 3, 1, 1))
        self.conv3 = (nn.Conv2d(4, 8, 3, 1, 1))
        self.conv4 = (nn.Conv2d(8, 8, 3, 1, 1))
        self.conv5 = (nn.Conv2d(2, 4, 3, 1, 1))
        self.conv6 = (nn.Conv2d(4, 1, 3, 1, 1))
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(-1, 8, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        x = torch.sigmoid(x)
        return 1 - x.view(-1, 1, 64, 64)

class Shaper64Noop(nn.Module):
    def __init__(self):
        super(Shaper64Noop, self).__init__()
        self.fc1 = (nn.Linear(9, 512))
        self.fc2 = (nn.Linear(512, 1024))
        self.fc3 = (nn.Linear(1024, 2048))
        self.conv1 = (nn.Conv2d(8, 16, 3, 1, 1))
        self.conv2 = (nn.Conv2d(16, 16, 3, 1, 1))
        self.conv3 = (nn.Conv2d(4, 8, 3, 1, 1))
        self.conv4 = (nn.Conv2d(8, 8, 3, 1, 1))
        self.conv5 = (nn.Conv2d(2, 4, 3, 1, 1))
        self.conv6 = (nn.Conv2d(4, 1, 3, 1, 1))
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(-1, 8, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        x = torch.sigmoid(x)
        return 1 - x.view(-1, 1, 64, 64)


class Shaper128(nn.Module):
    def __init__(self):
        super(Shaper128, self).__init__()
        self.fc1 = (nn.Linear(8, 512))
        self.fc2 = (nn.Linear(512, 1024))
        self.fc3 = (nn.Linear(1024, 2048))
        self.fc4 = (nn.Linear(2048, 4096))
        self.conv1 = (nn.Conv2d(16, 32, 3, 1, 1))
        self.conv2 = (nn.Conv2d(32, 32, 3, 1, 1))
        self.conv3 = (nn.Conv2d(8, 16, 3, 1, 1))
        self.conv4 = (nn.Conv2d(16, 16, 3, 1, 1))
        self.conv5 = (nn.Conv2d(4, 8, 3, 1, 1))
        self.conv6 = (nn.Conv2d(8, 4, 3, 1, 1))
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))
        x = torch.sigmoid(x)
        return 1 - x.view(-1, 1, 128, 128)

class Shaper128Noop(nn.Module):
    def __init__(self):
        super(Shaper128Noop, self).__init__()
        self.fc1 = (nn.Linear(9, 512))
        self.fc2 = (nn.Linear(512, 1024))
        self.fc3 = (nn.Linear(1024, 2048))
        self.fc4 = (nn.Linear(2048, 4096))
        self.conv1 = (nn.Conv2d(16, 32, 3, 1, 1))
        self.conv2 = (nn.Conv2d(32, 32, 3, 1, 1))
        self.conv3 = (nn.Conv2d(8, 16, 3, 1, 1))
        self.conv4 = (nn.Conv2d(16, 16, 3, 1, 1))
        self.conv5 = (nn.Conv2d(4, 8, 3, 1, 1))
        self.conv6 = (nn.Conv2d(8, 4, 3, 1, 1))
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))
        x = torch.sigmoid(x)
        return 1 - x.view(-1, 1, 128, 128)


class Drawer:
    def __init__(self, args, device):
        minf = str(int(args.wmin*10)).zfill(2)
        maxf = str(int(args.wmax*10)).zfill(2)
        if args.scale == 'small':
            if args.noop:
                self.shaper = Shaper64Noop().to(device)
            else:
                self.shaper = Shaper64().to(device)
            self.shaper.load_state_dict(torch.load(f'weights/shapers/shaper{minf}{maxf}snoop.pkl'))
        elif args.scale == 'medium':
            if args.noop:
                self.shaper = Shaper128Noop().to(device)
            else:
                self.shaper = Shaper128().to(device)
            self.shaper.load_state_dict(torch.load(f'weights/shapers/shaper{minf}{maxf}mnoop.pkl'))
        else:
            print('not implemented yet')
            exit()
    
    # noop
    def draw_nondiff(self, canvas, source, action, width, height, wmin, wmax, hmin, hmax, device, rounding=False, paper_like=False, shape=None):
        if shape is not None:
            mask = shape
        else:
            # shape
            x, y = (action[:, 1:2]-0.5)*width, (action[:, 2:3]-0.5)*height
            w = (action[:, 3:4]*(wmax-wmin)+wmin)*width
            h = (action[:, 4:5]*(hmax-hmin)+hmin)*height
            x *= (width-w)/(width)
            y *= (height-h)/(height)
            sx1, sy1 = width/2 - w/2 + w*action[:, 5:6], height/2 + h/2
            sx2, sy2 = width/2 + w/2, height/2 - h/2 + h*action[:, 8:9]
            sx3, sy3 = width/2 - w/2 + w*action[:, 6:7], height/2 - h/2
            sx4, sy4 = width/2 - w/2, height/2 - h/2 + h*action[:, 7:8]

            sp1 = torch.cat([sx1, sy1], 1)
            sp2 = torch.cat([sx2, sy2], 1)
            sp3 = torch.cat([sx3, sy3], 1)
            sp4 = torch.cat([sx4, sy4], 1)

            base = torch.zeros([action.size(0), 1, height, width]).to(device)
            polygons = torch.stack([sp1, sp2, sp3, sp4], 1)
            if rounding:
                polygons = torch.round(polygons)
            colors = torch.ones([action.size(0), 1]).to(device)
            shape = kornia.utils.draw_convex_polygon(base, polygons, colors)
            
            # mask translation
            tm = torch.cat([x, y], 1)
            if rounding:
                tm = torch.round(tm)
            mask = kornia.geometry.transform.translate(shape, tm)

            # noop
            mask = mask * (action[:, 0:1] > 0.5).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, height, width)

            # cut
            piece = source * mask
            
            # pasting translation & rotation
            px = (action[:, 9:10]-0.5)*width - x
            py = (action[:, 10:11]-0.5)*height - y
            pr = action[:, 11] * 180 - 90
            pt = torch.cat([px, py], 1)
            if rounding:
                pt = torch.round(pt)
                pr = torch.round(pr)
            ptrm = kornia.geometry.transform.get_affine_matrix2d(translations=pt,
                                                                center=torch.FloatTensor([width/2, height/2]).unsqueeze(0).repeat(canvas.size(0), 1).to(device),
                                                                scale=torch.ones([canvas.size(0), 2]).to(device),
                                                                angle=pr)
            transformed_piece = kornia.geometry.warp_affine(piece, ptrm[:, :2, :], (width, height))
            transformed_mask = kornia.geometry.warp_affine(mask, ptrm[:, :2, :], (width, height))

        # paper effect (looks like teared paper)
        if paper_like:
            translate_factor = torch.rand([canvas.size(0), 2]).to(device)*(1/100)*canvas.size(-1)
            paper_mask = kornia.geometry.transform.translate(transformed_mask, translate_factor, padding_mode='zeros')
            transformed_piece = transformed_piece * transformed_mask + paper_mask * (1 - transformed_mask)

        next_canvas = canvas*(1-transformed_mask) + transformed_piece

        return next_canvas, transformed_mask[:, 0, :, :].detach()

    # noop
    def draw_diff(self, canvas, source, action, width, height, wmin, wmax, hmin, hmax, device, rounding=False, paper_like=False, shape=None):
        if shape is not None:
            mask = shape
        else:
            mask = self.shaper(action[:, :9])

            # cut
            piece = source * mask
            
            # pasting translation & rotation
            x, y = (action[:, 1:2]-0.5)*width, (action[:, 2:3]-0.5)*height
            w = (action[:, 3:4]*(wmax-wmin)+wmin)*width
            h = (action[:, 4:5]*(hmax-hmin)+hmin)*height
            x *= (width-w)/(width)
            y *= (height-h)/(height)
            px = (action[:, 9:10]-0.5)*width - x
            py = (action[:, 10:11]-0.5)*height - y
            pr = action[:, 11] * 180 - 90
            pt = torch.cat([px, py], 1)
            ptrm = kornia.geometry.transform.get_affine_matrix2d(translations=pt,
                                                                center=torch.FloatTensor([width/2, height/2]).unsqueeze(0).repeat(canvas.size(0), 1).to(device),
                                                                scale=torch.ones([canvas.size(0), 2]).to(device),
                                                                angle=pr)
            transformed_piece = kornia.geometry.warp_affine(piece, ptrm[:, :2, :], (width, height))
            transformed_mask = kornia.geometry.warp_affine(mask, ptrm[:, :2, :], (width, height))

        # paper effect (looks like teared paper)
        if paper_like:
            translate_factor = torch.rand([canvas.size(0), 2]).to(device)*(1/100)*canvas.size(-1)
            paper_mask = kornia.geometry.transform.translate(transformed_mask, translate_factor, padding_mode='zeros')
            transformed_piece = transformed_piece * transformed_mask + paper_mask * (1 - transformed_mask)

        next_canvas = canvas*(1-transformed_mask) + transformed_piece

        return next_canvas, transformed_mask[:, 0, :, :].detach()
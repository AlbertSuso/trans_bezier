import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''ResNet Block'''

    def __init__(self, in_size, out_size, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.conv3 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.conv_res = nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_res = nn.BatchNorm2d(out_size)

    def forward(self, x, maxpool=True):
        residual = x
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x + self.bn_res(self.conv_res(residual))
        if maxpool:
            x = F.max_pool2d(nn.ReLU()(x), 2)
        return x


class NumCP_predictor(nn.Module):
    ''' ResNet used to predict the number of CP of a bezier curve'''

    def __init__(self, max_cp=6, in_chanels=1):
        super(NumCP_predictor, self).__init__()
        self.max_cp = max_cp

        self.block1 = Block(in_chanels, 32)
        self.block2 = Block(32, 64)
        self.block3 = Block(64, 128)
        self.block4 = Block(128, 256)
        self.fc1 = nn.Linear(256*8*8, 200)
        self.fc2 = nn.Linear(200, max_cp-1)

    def forward(self, input):
        # Input 1x64x64
        x = self.block1(input)
        # Input 32x32x32
        x = self.block2(x)
        # Input 64x16x16
        x = self.block3(x)
        # Input 128x8x8
        x = self.block4(x, maxpool=False)
        return self.fc2(F.relu(self.fc1(x.view(input.shape[0], -1))))

class ResNet12(nn.Module):
    ''' ResNet used to predict the number of CP of a bezier curve'''

    def __init__(self, max_cp=6, in_chanels=1):
        super(ResNet12, self).__init__()
        self.d_model = 256

        self.block1 = Block(in_chanels, 32)
        self.block2 = Block(32, 64)
        self.block3 = Block(64, 128)
        self.block4 = Block(128, 256)

    def forward(self, input):
        # Input 1x64x64
        x = self.block1(input)
        # Input 32x32x32
        x = self.block2(x)
        # Input 64x16x16
        x = self.block3(x)
        # Input 128x8x8
        x = self.block4(x, maxpool=False)
        return x.view(input.shape[0], self.d_model, -1).permute(2, 0, 1)

class ResNet18(nn.Module):
    def __init__(self, in_chanels=1):
        super(ResNet18, self).__init__()
        self.block1 = Block(in_chanels, 16)
        self.block2 = Block(16, 32)
        self.block3 = Block(32, 64)
        self.block4 = Block(64, 128)
        self.block5 = Block(128, 256)
        self.block6 = Block(256, 512)
        self.d_model = 512

    def forward(self, input):
        # Input 64x64x1
        x = self.block1(input)
        # Input 32x32x16
        x = self.block2(x)
        # Input 16x16x32
        x = self.block3(x)
        # Input 8x8x64
        x = self.block4(x)
        # Input 4x4x128
        x = self.block5(x)
        # Input 2x2x256
        x = self.block6(x, maxpool=False)
        return  x.view(input.shape[0], self.d_model, -1).permute(2, 0, 1) #La pasamos a la shape adecuada que necesita la transformer # MIRAR PERMUTE!!!
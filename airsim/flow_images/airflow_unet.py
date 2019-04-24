import sys
sys.path.append("/home/dhruvkar/Desktop/Robotics/rp/Airflownet/src")
import torch
import torch.nn as nn
import torch.nn.functional as F


#256 x 256 Airflow Images
class Airflow_Unet256(nn.Module):
    def __init__(self, in_shape, sdf):
        super(Airflow_Unet256, self).__init__()
        self.sdf = sdf
        C, H, W = in_shape
        #256
        self.down1 = StackEncoder(C, 64, kernel_size=3) #128
        self.down2 = StackEncoder(64, 128, kernel_size=3) #64
        self.down3 = StackEncoder(128, 256, kernel_size=3) #32
        self.down4 = StackEncoder(256, 512, kernel_size=3) #16
        self.down5 = StackEncoder(512, 1024, kernel_size=3) #8
        
        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1)
        )
        #8
        self.up5 = StackDecoder(1024, 1024, 512, kernel_size=3) #16
        self.up4 = StackDecoder(512, 512, 256, kernel_size=3) #32
        self.up3 = StackDecoder(256, 256, 128, kernel_size=3) #64
        self.up2 = StackDecoder(128, 128, 64, kernel_size=3) #128
        self.up1 = StackDecoder(64, 64, 24, kernel_size=3) #64
        self.classify = nn.Conv2d(24, 1, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        if self.sdf:
            mask = torch.where(x <= 0, torch.zeros_like(x), torch.ones_like(x))
        else:
            mask = x
        out = x

        down1, out = self.down1(out)

        down2, out = self.down2(out)

        down3, out = self.down3(out)

        down4, out = self.down4(out)

        down5, out = self.down5(out)

        out = self.center(out)

        out = self.up5(down5, out)

        out = self.up4(down4, out)

        out = self.up3(down3, out)

        out = self.up2(down2, out)

        out = self.up1(down1, out)

        out = self.classify(out)
        out = out * mask
        out = torch.squeeze(out, dim=1)
        return out

class StackDecoder(nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super(StackDecoder, self).__init__()
        padding = (kernel_size - 1)//2

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),

            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),

            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1)
        )
    def forward(self,x_big, x):
        #First upsample the output 
        N, C, H, W = x_big.size()
        y = F.upsample(x, size=(H, W), mode='bilinear')
        y = torch.cat([y, x_big], 1) #This step concatenates the initial image to upsampled image (Skip Connection)
        y = self.decode(y)
        return y


#Encapsulates a single Stack of down operations, including ConvBnRelu and MaPool
class StackEncoder(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3):
        super(StackEncoder, self).__init__()
        padding = (kernel_size -1) // 2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1),
            
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1, groups=1)
        )

    def forward(self, x):
        y = self.encode(x)
        y_pool = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_pool


BN_EPS = 1e-4

#Apply a Convolutional Layer, Batchnorm and Relu in one go
class ConvBnRelu2d(nn.Module):
    
    #Define Net Structure for ease of use
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True, 
    is_relu=True):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)
        if not is_bn: self.bn = None
        if not is_relu: self.relu = None

    def forward(self, x):
        x = self.conv(x) #nn.Conv2d ...
        if self.bn is not None:
            x = self.bn(x)

        if self.relu is not None:
            x = self.relu(x)

        return x
import torch
import torch.nn.functional as F
from torch.nn.init import xavier_uniform

import logs
info_logger, data_logger = logs.get_loggers()

class GuoCNN(torch.nn.Module):
    def __init__(self, sdf):
        super(GuoCNN, self).__init__()

        self.sdf = sdf

        self.conv1 = torch.nn.Conv2d(1, 128, kernel_size=(16, 16), stride=(16, 16))
        xavier_uniform(self.conv1.weight)
        self.conv2 = torch.nn.Conv2d(128, 512, kernel_size=(4, 4), stride=(4, 4))
        xavier_uniform(self.conv2.weight)

        self.fc1 = torch.nn.Linear(4 * 4 * 512, 1024)
        xavier_uniform(self.fc1.weight)

        self.deconv1 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=(8, 8), stride=(8, 8))
        xavier_uniform(self.deconv1.weight)
        self.deconv2 = torch.nn.ConvTranspose2d(512, 256, kernel_size=(8, 8), stride=(8, 8))
        xavier_uniform(self.deconv2.weight)
        self.deconv3 = torch.nn.ConvTranspose2d(256, 32, kernel_size=(2, 2), stride=(2, 2))
        xavier_uniform(self.deconv3.weight)
        self.deconv4 = torch.nn.ConvTranspose2d(32, 1, kernel_size=(2, 2), stride=(2, 2))
        xavier_uniform(self.deconv4.weight)

    def forward(self, x):
        if self.sdf:
            mask = torch.where(x <= 0, torch.zeros_like(x), torch.ones_like(x))
        else:
            mask = x

        activation = F.relu

        x = activation(self.conv1(x))
        x = activation(self.conv2(x))

        x = x.view(-1, 4 * 4 * 512)
        x = activation(self.fc1(x))

        x = x.view(-1, 1024, 1, 1)
        x = activation(self.deconv1(x))
        x = activation(self.deconv2(x))
        x = activation(self.deconv3(x))
        x = self.deconv4(x)

        x = x * mask
        return x

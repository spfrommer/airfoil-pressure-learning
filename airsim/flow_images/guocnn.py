import torch
import torch.nn.functional as F
from torch.nn.init import xavier_uniform

import logs
info_logger, data_logger = logs.get_training_loggers()

class GuoCNN(torch.nn.Module):
    def __init__(self):
        super(GuoCNN, self).__init__()

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
        # Prevents zeros in last deconvolution output from bias being negative
        self.deconv4.bias.data = torch.tensor([0.1])

    def forward(self, x):
        mask = x.clone()

        info_logger.debug("Forwards pass")
        info_logger.debug(x.size())

        x = F.relu(self.conv1(x))
        info_logger.debug("After conv1")
        info_logger.debug(x.size())

        x = F.relu(self.conv2(x))
        info_logger.debug("After conv2")
        info_logger.debug(x.size())

        x = x.view(-1, 4 * 4 * 512)
        x = F.relu(self.fc1(x))
        info_logger.debug("After fc1")
        info_logger.debug(x.size())

        x = x.view(-1, 1024, 1, 1)
        x = F.relu(self.deconv1(x))
        info_logger.debug("After deconv1")
        info_logger.debug(x.size())

        x = F.relu(self.deconv2(x))
        info_logger.debug("After deconv2")
        info_logger.debug(x.size())
        #if x.sum().item() < 0.00000001:
            #info_logger.warn("All zeros!")

        x = F.relu(self.deconv3(x))
        info_logger.debug("After deconv3")
        info_logger.debug(x.size())
        #if x.sum().item() < 0.00000001:
            #info_logger.warn("All zeros!")

        x = F.relu(self.deconv4(x))
        info_logger.debug("After deconv4")
        info_logger.debug(x.size())
        #if x.sum().item() < 0.00000001:
            #info_logger.warn("All zeros!")

        x = x * mask
        return x

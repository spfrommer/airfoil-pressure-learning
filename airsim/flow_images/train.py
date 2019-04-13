import sys
import os
import os.path as op
path = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
print('Setting project root path: ' + path)
sys.path.append(path)

import time
import re
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.autograd.variable import Variable
from torch.nn.init import xavier_uniform
from torchvision import transforms, utils
import torch.nn.functional as F
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import airsim.dirs as dirs
import logs

info_logger, data_logger = logs.create_training_loggers()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sdf_samples = True
test_trained_net = True
load_net = True
net_path = dirs.out_path('trained', 'net.pth')
epochs = 0
batch_size = 64
learning_rate = 0.0001

class BinaryPressureDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        
        files = os.listdir(root_path)
        nums = list(dict.fromkeys([re.sub('\D','',f) for f in files]))

        #samples = np.repeat(nums, 1)
        #transx = np.round(0 * (np.random.rand(len(samples),1) - 0.3))
        #transy = np.round(0 * (np.random.rand(len(samples),1) - 0.5))
        #flip = np.tile([0], len(samples))
        samples = np.repeat(nums, 10)
        transx = np.round(80 * (np.random.rand(len(samples),1) - 0.1))
        transy = np.round(110 * (np.random.rand(len(samples),1) - 0.5))
        flip = np.tile([0,1], len(samples)//2)
        self.samples = zip(samples, transx, transy, flip)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        
        if sdf_samples:
            airfoil_file = 'sdf_{}.pt'.format(sample[0]) 
        else:
            airfoil_file = 'a_{}.pt'.format(sample[0]) 

        pressure_file = 'p_{}.pt'.format(sample[0]) 
        airfoil = torch.load(op.join(self.root_path, airfoil_file)).to(device)
        pressure = torch.load(op.join(self.root_path, pressure_file)).to(device)

        centerx = int((airfoil.size()[1] // 2 + sample[1])[0])
        centery = int((airfoil.size()[0] // 2 + sample[2])[0])
        half_size = 128
       
        airfoil = airfoil[centery - half_size : centery + half_size,
                          centerx - half_size : centerx + half_size]
        pressure = pressure[centery - half_size : centery + half_size,
                            centerx - half_size : centerx + half_size]

        if sample[3] == 1:
            airfoil = torch.flip(airfoil, [0])
            pressure = torch.flip(pressure, [0])

        # Add dummy channel
        airfoil = airfoil.expand(1,-1,-1)
        pressure = pressure.expand(1,-1,-1)

        return (airfoil, pressure)

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

def airfoilmseloss(pred, actual):
    foilmask = torch.where(actual == -1000, 0, 1)
    pred = foilmask * pred
    actual = foilmask * actual
    
    diff = pred - actual
    diffsq = torch.pow(diff, 2)
    return torch.mean(diffsq)

train_dataset = BinaryPressureDataset(dirs.out_path('processed', 'train'))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

test_dataset = BinaryPressureDataset(dirs.out_path('processed', 'test'))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

info_logger.info("Training dataset size: {}".format(len(train_dataset)))
info_logger.info("Testing dataset size: {}".format(len(test_dataset)))

net = GuoCNN().to(device)
if load_net:
    net.load_state_dict(torch.load(net_path))

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

batches = len(train_loader)
training_start_time = time.time()

for epoch in range(epochs):
    running_loss = 0.0
    print_every = batches // 5
    start_time = time.time()
    total_train_loss = 0
    
    for i, data in enumerate(train_loader, 0):
        airfoils, pressures = data
        airfoils, pressures = Variable(airfoils), Variable(pressures)
        
        optimizer.zero_grad()
        
        #Forward pass, backward pass, optimize
        pressures_pred = net(airfoils)
        loss_size = loss(pressures_pred, pressures)
        loss_size.backward()
        optimizer.step()
        
        #Print statistics
        running_loss += loss_size.item()
        total_train_loss += loss_size.item()

        #Print every nth batch of an epoch
        if (i + 1) % (print_every + 1) == 0:
            info_logger.info("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch+1, int(100 * (i+1) / batches), running_loss / print_every, time.time() - start_time))
            #Reset running loss and time
            running_loss = 0.0
            start_time = time.time()
        
    #At the end of the epoch, do a pass on the test set
    total_test_loss = 0
    for airfoils, pressures in test_loader:
        airfoils, pressures = Variable(airfoils), Variable(pressures)

        pressures_pred = net(airfoils)
        loss_size = loss(pressures_pred, pressures)
        total_test_loss += loss_size.item()
       
    avg_train_loss = total_train_loss / len(train_dataset)
    avg_test_loss = total_test_loss / len(test_dataset)
    info_logger.info("Finished epoch {}".format(epoch+1))
    info_logger.info("Train loss = {:.2f}".format(avg_train_loss))
    info_logger.info("Test loss = {:.2f}".format(avg_test_loss))
    data_logger.info("{}, {}, {}".format(epoch, avg_train_loss, avg_test_loss))
    
info_logger.info("Training finished, took {:.2f}s".format(time.time() - training_start_time))

torch.save(net.state_dict(), net_path)

if test_trained_net:
    airfoil, pressure = test_dataset[0]
    airfoil, pressure = Variable(airfoil), Variable(pressure)
    airfoil = airfoil.view(1, 1, 256, 256)
    pressure = pressure.view(1, 1, 256, 256)
    pressure_pred = net(airfoil)
    pressure_error = pressure_pred - pressure

    info_logger.info(airfoil)
    utils.save_image(airfoil, 'airfoil.png')

    info_logger.info(pressure)
    pressure = (pressure / 400) + 0.5
    utils.save_image(pressure, 'pressure.png')

    info_logger.info(pressure_pred)
    pressure_pred = (pressure_pred / 200) + 0.5
    utils.save_image(pressure_pred, 'pressure_pred.png')

    info_logger.info(pressure_error)
    pressure_error = (pressure_error / 200) + 0.5
    utils.save_image(pressure_error, 'pressure_error.png')



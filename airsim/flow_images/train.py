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
from torchvision import transforms, utils
import torch.nn.functional as F
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import airsim.dirs as dirs
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
        transx = np.round(100 * (np.random.rand(len(samples),1) - 0.3))
        transy = np.round(20 * (np.random.rand(len(samples),1) - 0.5))
        flip = np.tile([0,1], len(samples)/2)
        self.samples = zip(samples, transx, transy, flip)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]

        airfoil_file = 'a_{}.pt'.format(sample[0]) 
        pressure_file = 'p_{}.pt'.format(sample[0]) 
        airfoil = torch.load(op.join(self.root_path, airfoil_file))
        pressure = torch.load(op.join(self.root_path, pressure_file))

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

class PaperCNN(torch.nn.Module):
    def __init__(self):
        super(PaperCNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 128, kernel_size=(16, 16), stride=(16, 16))
        self.conv2 = torch.nn.Conv2d(128, 512, kernel_size=(4, 4), stride=(4, 4))

        self.fc1 = torch.nn.Linear(4 * 4 * 512, 1024)

        self.deconv1 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=(8, 8), stride=(8, 8))
        self.deconv2 = torch.nn.ConvTranspose2d(512, 256, kernel_size=(8, 8), stride=(8, 8))
        self.deconv3 = torch.nn.ConvTranspose2d(256, 32, kernel_size=(2, 2), stride=(2, 2))
        self.deconv4 = torch.nn.ConvTranspose2d(32, 1, kernel_size=(2, 2), stride=(2, 2))
        #print(self.deconv4.weight.size())
        #print(self.deconv4.weight.data)


    def forward(self, x):
        print("Forwards pass")
        print(x.size())
        #print x
        x = F.relu(self.conv1(x))
        print("After conv1")
        print(x.size())
        #print x
        x = F.relu(self.conv2(x))
        print("After conv2")
        print(x.size())
        #print x
        x = x.view(-1, 4 * 4 * 512)
        x = F.relu(self.fc1(x))
        print("After fc1")
        print(x.size())
        #print x
        x = x.view(-1, 1024, 1, 1)
        x = F.relu(self.deconv1(x))
        print("After deconv1")
        print(x.size())
        #print x
        x = F.relu(self.deconv2(x))
        print("After deconv2")
        print(x.size())
        if x.sum().item() < 0.00000001:
            print("All zeros!")
        #print x
        x = F.relu(self.deconv3(x))
        print("After deconv3")
        print(x.size())
        if x.sum().item() < 0.00000001:
            print("All zeros!")
        #print x
        x = F.relu(self.deconv4(x))
        print("After deconv4")
        print(x.size())
        if x.sum().item() < 0.00000001:
            print("All zeros!")
        return x

def airfoilmseloss(pred, actual):
    foilmask = torch.where(actual == -1000, 0, 1)
    pred = foilmask * pred
    actual = foilmask * actual
    
    diff = pred - actual
    diffsq = torch.pow(diff, 2)
    return torch.mean(diffsq)

epochs = 1
batch_size = 4
learning_rate = 0.0001
train_dataset = BinaryPressureDataset(dirs.out_path('processed', 'train'))
train_sampler = RandomSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

test_dataset = BinaryPressureDataset(dirs.out_path('processed', 'test'))
test_sampler = RandomSampler(test_dataset)
test_loader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler)

print(train_dataset.__len__())

net = PaperCNN()
# TODO: ignore airfoil mask bits
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

batches = len(train_loader)
training_start_time = time.time()

for epoch in range(epochs):
    running_loss = 0.0
    print_every = batches // 10
    start_time = time.time()
    total_train_loss = 0
    
    for i, data in enumerate(train_loader, 0):
        airfoils, pressures = data
        airfoils, pressures = Variable(airfoils), Variable(pressures)
        
        optimizer.zero_grad()
        
        print("Training")
        #Forward pass, backward pass, optimize
        pressures_pred = net(airfoils)
        loss_size = loss(pressures_pred, pressures)
        loss_size.backward()
        optimizer.step()
        
        #Print statistics
        running_loss += loss_size.item()
        total_train_loss += loss_size.item()
        
        #Print every 10th batch of an epoch
        if (i + 1) % (print_every + 1) == 0:
            print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch+1, int(100 * (i+1) / batches), running_loss / print_every, time.time() - start_time))
            #Reset running loss and time
            running_loss = 0.0
            start_time = time.time()
        
    #At the end of the epoch, do a pass on the test set
    total_test_loss = 0
    for airfoils, pressures in test_loader:
        airfoils, pressures = Variable(airfoils), Variable(pressures)
        
        print("Testing")
        test_pressures_pred = net(airfoils)
        test_loss_size = loss(test_pressures_pred, pressures)
        total_test_loss += test_loss_size.item()
        
    print("Test loss = {:.2f}".format(total_test_loss / len(test_loader)))
    
print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

airfoil, pressure = test_dataset[0]
airfoil, pressure = Variable(airfoil), Variable(pressure)
airfoil = airfoil.view(1, 1, 256, 256)
pressure = pressure.view(1, 1, 256, 256)
pressure_pred = net(airfoil)

print(airfoil)
utils.save_image(airfoil, 'airfoil.png')

print(pressure)
pressure = (pressure / 400) + 0.5
utils.save_image(pressure, 'pressure.png')

print(pressure_pred)
#pressure_pred = (pressure_pred / 200) + 0.5
utils.save_image(pressure_pred, 'pressure_pred.png')

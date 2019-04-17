import sys
import os
import os.path as op
path = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
print('Setting project root path: ' + path)
sys.path.append(path)

import time
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.autograd.variable import Variable
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import logs
info_logger, data_logger = logs.create_training_loggers()
from guocnn import GuoCNN
from dataset import ProcessedAirfoilDataset

import airsim.dirs as dirs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sdf_samples = False
net_path = dirs.out_path('trained', 'net.pth')
batch_size = 64

def main():
    #setup_multiprocessing()
    train_dataset, train_loader, test_dataset, test_loader = init_data()
    net = GuoCNN().to(device)
    net.load_state_dict(torch.load(net_path, map_location=device))
    print net
    test_net(net, test_dataset[0]) 

def setup_multiprocessing():
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

def init_data():
    train_dataset = ProcessedAirfoilDataset(
            dirs.out_path('processed', 'train'), sdf_samples, device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    test_dataset = ProcessedAirfoilDataset(
            dirs.out_path('processed', 'test'), sdf_samples, device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0)
    info_logger.info("Training dataset size: {}".format(len(train_dataset)))
    info_logger.info("Testing dataset size: {}".format(len(test_dataset)))
    return train_dataset, train_loader, test_dataset, test_loader

def test_net(net, sample):
    airfoil, pressure = sample
    airfoil, pressure = Variable(airfoil), Variable(pressure)
    airfoil = airfoil.view(1, 1, 256, 256)
    pressure = pressure.view(1, 1, 256, 256)
    pressure_pred = net(airfoil)
    pressure_error = pressure_pred - pressure

    info_logger.info(airfoil)
    utils.save_image(airfoil, dirs.out_path('testing', 'airfoil.png'))

    info_logger.info(pressure)
    pressure = pressure + 0.5
    utils.save_image(pressure, dirs.out_path('testing', 'pressure.png'))

    info_logger.info(pressure_pred)
    pressure_pred = pressure_pred + 0.5
    utils.save_image(pressure_pred, dirs.out_path('testing', 'pressure_pred.png'))

    info_logger.info(pressure_error)
    pressure_error = pressure_error + 0.5
    utils.save_image(pressure_error, dirs.out_path('testing', 'pressure_error.png'))

if __name__ == "__main__":main()


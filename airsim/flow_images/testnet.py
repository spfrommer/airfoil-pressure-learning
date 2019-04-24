import sys
import os
import os.path as op
path = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
print('Setting project root path: ' + path)
sys.path.append(path)

import time
import torch
from torch.autograd.variable import Variable
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import logs
info_logger, data_logger = logs.create_loggers(training=False)
from guocnn import GuoCNN
import dataset

import airsim.dirs as dirs
from airsim.io_utils import empty_dir

empty_dir(dirs.out_path('testing'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sdf_samples = False
net_path = dirs.out_path('training', 'final_net.pth')
batch_size = 1

def main():
    #setup_multiprocessing()
    (train_dataset, train_loader, validation_dataset, validation_loader,
            test_dataset, test_loader) = dataset.load_data(sdf_samples, device, batch_size, 0)
    #net = GuoCNN(sdf_samples).to(device)
    net.load_state_dict(torch.load(net_path, sdf_samples, map_location=device))
    sample = train_dataset[0]
    info_logger.info("Input largest element: {}".format(torch.max(sample[0])))
    info_logger.info("Input smallest element: {}".format(torch.min(sample[0])))
    info_logger.info("Pressure largest element: {}".format(torch.max(sample[1])))
    info_logger.info("Pressure smallest element: {}".format(torch.min(sample[1])))
    test_net(net, sample) 

def setup_multiprocessing():
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

def test_net(net, sample):
    airfoil, pressure = sample
    airfoil, pressure = Variable(airfoil), Variable(pressure)
    airfoil = airfoil.view(1, 1, 256, 256)
    pressure = pressure.view(1, 1, 256, 256)
    pressure_pred = net(airfoil)
    pressure_error = pressure_pred - pressure

    utils.save_image(airfoil, dirs.out_path('testing', 'airfoil.png'))

    pressure = pressure + 0.5
    utils.save_image(pressure, dirs.out_path('testing', 'pressure.png'))

    pressure_pred = pressure_pred + 0.5
    utils.save_image(pressure_pred, dirs.out_path('testing', 'pressure_pred.png'))

    pressure_error = pressure_error + 0.5
    utils.save_image(pressure_error, dirs.out_path('testing', 'pressure_error.png'))

if __name__ == "__main__":main()

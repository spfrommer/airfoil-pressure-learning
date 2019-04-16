import os
import os.path as op
import torch
from torch.utils.data import Dataset
import re
import numpy as np

import logs
info_logger, data_logger = logs.get_training_loggers()

class ProcessedAirfoilDataset(Dataset):
    def __init__(self, root_path, sdf_samples, device):
        self.root_path = root_path
        self.sdf_samples = sdf_samples
        self.device = device
        
        files = os.listdir(root_path)
        nums = list(dict.fromkeys([re.sub('\D','',f) for f in files]))

        samples = np.repeat(nums, 10)
        transx = np.round(80 * (np.random.rand(len(samples),1) - 0.1))
        transy = np.round(110 * (np.random.rand(len(samples),1) - 0.5))
        flip = np.tile([0,1], len(samples)//2)
        self.samples = list(zip(samples, transx, transy, flip))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        
        if self.sdf_samples:
            airfoil_file = 'sdf_{}.pt'.format(sample[0]) 
        else:
            airfoil_file = 'a_{}.pt'.format(sample[0]) 

        pressure_file = 'p_{}.pt'.format(sample[0]) 
        airfoil = torch.load(op.join(self.root_path, airfoil_file)).to(self.device)
        pressure = torch.load(op.join(self.root_path, pressure_file)).to(self.device)

        centerx = int((airfoil.size()[1] // 2 + sample[1])[0])
        centery = int((airfoil.size()[0] // 2 + sample[2])[0])
        half_size = 128
       
        airfoil = airfoil[centery - half_size : centery + half_size,
                          centerx - half_size : centerx + half_size]
        pressure = pressure[centery - half_size : centery + half_size,
                            centerx - half_size : centerx + half_size]
        pressure = pressure / 400
        if sample[3] == 1:
            airfoil = torch.flip(airfoil, [0])
            pressure = torch.flip(pressure, [0])

        # Add dummy channel
        airfoil = airfoil.expand(1,-1,-1)
        pressure = pressure.expand(1,-1,-1)

        return (airfoil, pressure)


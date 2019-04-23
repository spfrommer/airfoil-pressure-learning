import os
import os.path as op
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import re
import numpy as np

from airsim import dirs
import logs
info_logger, data_logger = logs.get_loggers()

class ProcessedAirfoilDataset(Dataset):
    def __init__(self, root_path, sdf_samples, device, augment=True):
        self.root_path = root_path
        self.sdf_samples = sdf_samples
        self.device = device
        
        files = os.listdir(root_path)
        nums = list(dict.fromkeys([re.sub('\D','',f) for f in files]))

        if augment:
            samples = np.repeat(nums, 10)
            transx = np.round(80 * (np.random.rand(len(samples),1) - 0.1))
            transy = np.round(110 * (np.random.rand(len(samples),1) - 0.5))
            flip = np.tile([0,1], len(samples)//2)
            self.samples = list(zip(samples, transx, transy, flip))
        else:
            samples = nums
            #transx = np.zeros((len(samples),1))
            transx = np.ones((len(samples),1)) * 40
            transy = np.zeros((len(samples),1))
            flip = np.zeros((len(samples),1))
            self.samples = list(zip(samples, transx, transy, flip))

    def __len__(self):
        return len(self.samples)
    
    def load_tensor(self, filename_root, sample):
        tensor = torch.load(op.join(self.root_path,
            '{}_{}.pt'.format(filename_root, sample[0]))).to(self.device)
        return tensor

    def __getitem__(self, sample_id):
        sample = self.samples[sample_id]
        
        airfoil = self.load_tensor('sdf' if self.sdf_samples else 'a', sample)
        pressure = self.load_tensor('p', sample)
        centerx = int((airfoil.size()[1] // 2 + sample[1])[0])
        centery = int((airfoil.size()[0] // 2 + sample[2])[0])
        half_size = 128
       
        airfoil = airfoil[centery - half_size : centery + half_size,
                          centerx - half_size : centerx + half_size]
        pressure = pressure[centery - half_size : centery + half_size,
                            centerx - half_size : centerx + half_size]

        # Empirical tuning
        if self.sdf_samples: airfoil = (airfoil / 400.0)
        pressure = (pressure / 400.0)

        if sample[3] == 1:
            airfoil = torch.flip(airfoil, [0])
            pressure = torch.flip(pressure, [0])

        # Add dummy channel
        airfoil = airfoil.expand(1,-1,-1)
        pressure = pressure.expand(1,-1,-1)

        return airfoil, pressure, sample_id

def load_data(sdf_samples, device, batch_size, num_workers):
    train_dataset = ProcessedAirfoilDataset(
            dirs.out_path('processed', 'train'), sdf_samples, device, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    validation_dataset = ProcessedAirfoilDataset(
            dirs.out_path('processed', 'validation'), sdf_samples, device, augment=False)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_dataset = ProcessedAirfoilDataset(
            dirs.out_path('processed', 'test'), sdf_samples, device, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    info_logger.info("Training dataset size: {}".format(len(train_dataset)))
    info_logger.info("Validation dataset size: {}".format(len(validation_dataset)))
    info_logger.info("Testing dataset size: {}".format(len(test_dataset)))
    return train_dataset, train_loader, validation_dataset, validation_loader, test_dataset, test_loader


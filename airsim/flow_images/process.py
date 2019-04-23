import sys
import os
import os.path as op
path = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
print('Setting project root path: ' + path)
sys.path.append(path)

import numpy as np
import random
import scipy
import scipy.ndimage

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from matplotlib import pyplot

from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import airsim.dirs as dirs
from airsim.io_utils import empty_dir

class AirfoilDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.image_folders = os.listdir(root_path)
        if '.DS_Store' in self.image_folders:
            self.image_folders.remove('.DS_Store')

    def __len__(self):
        return len(self.image_folders)

    def __getitem__(self, index):
        image = Image.open(op.join(self.root_path, self.image_folders[i], 'p.png'))
        metadata = self.image_folders[i].split("_")
        metadata = (metadata[0], float(metadata[1]), float(metadata[2]))
        return metadata, image


empty_dir(dirs.out_path('processed', 'train'))
empty_dir(dirs.out_path('processed', 'validation'))
empty_dir(dirs.out_path('processed', 'test'))

output_images = True
airfoil_dataset = AirfoilDataset(dirs.out_path('images_sample'))

#test_prefixes = ["s1223"]
#data_splits = np.array([0.60, 0.75, 1]) * len(airfoil_dataset)
data_splits = np.array([0.5, 0.7, 1]) * len(airfoil_dataset)

data_indices = range(len(airfoil_dataset))
random.shuffle(data_indices)

for j, i in enumerate(data_indices):
    print('Processing: {}'.format(j));
    metadata, sample = airfoil_dataset[i]

    #if metadata[0].startswith(tuple(test_prefixes)):
        #save_dir = 'test'
    #else:
        #save_dir = 'train'
    if j < data_splits[0]:
        save_dir = 'train'
    elif j < data_splits[1]:
        save_dir = 'validation'
    else:
        save_dir = 'test'

    size = (800,800)
    sample.thumbnail(size, Image.NEAREST)

    tensor = transforms.ToTensor()(sample)
    
    transx = 40
    transy = 0
    centerx = int(tensor.size()[2] // 2 + transx)
    centery = int(tensor.size()[1] // 2 + transy)
    half_size = 128
    tensor = tensor[:, centery - half_size : centery + half_size,
                       centerx - half_size : centerx + half_size]

    # Make airfoil tensor
    flattened = torch.sum(tensor, dim = 0) 
    airfoil_mask = torch.where(flattened == 0, flattened, torch.ones_like(flattened))
    torch.save(airfoil_mask, dirs.out_path('processed', save_dir, 'a_{}.pt'.format(i)))

    # Make sdf tensor
    binmask = airfoil_mask.numpy()
    binmask = binmask.astype(int)
    posfoil = scipy.ndimage.morphology.distance_transform_edt(binmask)
    binmaskcomp = 1 - binmask
    negfoil = scipy.ndimage.morphology.distance_transform_edt(binmaskcomp)
    sdf = np.subtract(posfoil, negfoil)
    sdf_mask = torch.tensor(sdf).float()
    torch.save(sdf_mask, dirs.out_path('processed', save_dir, 'sdf_{}.pt'.format(i)))
    
    # Make pressure tensor
    pressure_range = (-1000, 1000)
    range_diff = pressure_range[1] - pressure_range[0] 
    range_increment = range_diff / 256.0
    pressure_mask = (tensor[1, :, :] - 0.5) * range_diff + tensor[2, :, :] * range_increment
    # Zero out elements within airfoil
    pressure_mask = pressure_mask * airfoil_mask
    torch.save(pressure_mask, dirs.out_path('processed', save_dir, 'p_{}.pt'.format(i)))

    if output_images:
        sdf_mask = (sdf_mask / 500) + 0.5
        utils.save_image(sdf_mask, dirs.out_path('processed', save_dir, 'sdf_{}.png'.format(i)))

        pressure_mask = (pressure_mask / 200) + 0.5
        utils.save_image(pressure_mask, dirs.out_path('processed', save_dir, 'p_{}.png'.format(i)))

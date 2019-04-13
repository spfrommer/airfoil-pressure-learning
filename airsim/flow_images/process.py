import sys
import os
import os.path as op
path = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
print('Setting project root path: ' + path)
sys.path.append(path)

import torch
import numpy as np
import scipy
import scipy.ndimage
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import airsim.dirs as dirs

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

test_prefixes = ["goe5"]
airfoil_dataset = AirfoilDataset(dirs.out_path('images'))
for i in range(len(airfoil_dataset)):
    print('Processing: {}'.format(i));
    metadata, sample = airfoil_dataset[i]

    if metadata[0].startswith(tuple(test_prefixes)):
        save_dir = 'test'
    else:
        save_dir = 'train'
    #sample = transforms.CenterCrop(4096)(sample)

    size = (800,800)
    sample.thumbnail(size, Image.NEAREST)

    tensor = transforms.ToTensor()(sample)

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

    sdf_mask = (sdf_mask / 500) + 0.5
    utils.save_image(sdf_mask, dirs.out_path('processed', save_dir, 'sdf_{}.png'.format(i)))
    
    # Make pressure tensor
    pressure_range = (-1000, 1000)
    range_diff = pressure_range[1] - pressure_range[0] 
    range_increment = range_diff / 256.0
    pressure_mask = (tensor[1, :, :] - 0.5) * range_diff + tensor[2, :, :] * range_increment
    torch.save(pressure_mask, dirs.out_path('processed', save_dir, 'p_{}.pt'.format(i)))

    pressure_mask = (pressure_mask / 200) + 0.5
    utils.save_image(pressure_mask, dirs.out_path('processed', save_dir, 'p_{}.png'.format(i)))

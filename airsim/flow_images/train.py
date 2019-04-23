import sys
import os
import os.path as op
path = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
print('Setting project root path: ' + path)
sys.path.append(path)

import numpy as np
import time
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.autograd.variable import Variable
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import logs
import losses
from guocnn import GuoCNN
import dataset

import airsim.dirs as dirs
from airsim.io_utils import empty_dir

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

resume = False

sdf_samples = False
validation_net_path = dirs.out_path('training', 'validation_net.pth')
final_net_path = dirs.out_path('training', 'final_net.pth')

epochs = 250
num_workers = 0
batch_size = 1
learning_rate = 0.001 * (batch_size / 64.0)
#learning_rate_mul = 0.8
#learning_rate_mul_interval = 2000 # Number of descents per lr rescale

start_epoch = 0
validation_min = float('inf')
append = False
load_net_path = None

if resume:
    append = True
    load_net_path = final_net_path
    log_path = dirs.out_path('training', 'data.log') 
    array = np.loadtxt(open(log_path, "rb"), delimiter=",", skiprows=0)
    start_epoch = np.size(array, 0)
    validation_min = np.min(array[:,2])
else:
    empty_dir(dirs.out_path('training'))

def main():
    if num_workers > 0:
        setup_multiprocessing()

    info_logger, data_logger = logs.create_loggers(training=True, append=append)

    if resume:
        info_logger.info("Resuming training")
        info_logger.info("Start epoch: {}".format(start_epoch))
        info_logger.info("Best validation error: {}".format(validation_min))

    (train_dataset, train_loader, validation_dataset, validation_loader,
            test_dataset, test_loader) = dataset.load_data(sdf_samples, device, batch_size, num_workers)
    net = GuoCNN(sdf=sdf_samples).to(device)
    if load_net_path:
        net.load_state_dict(torch.load(load_net_path))
    loss = losses.foil_mse_loss
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    #optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate,
    #        eps=1e-4, weight_decay=0.05, momentum=0.9)

    training_start_time = time.time()
    try:
        train(net, optimizer, loss, train_loader, validation_loader)
    except Exception:
        info_logger.exception("Error in training")

    info_logger.info("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    test_loss = loss_pass(net, loss, test_loader)
    info_logger.info("Test loss = {:.12f}".format(test_loss))

def setup_multiprocessing():
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

info_logger, data_logger = logs.get_loggers()

def train(net, optimizer, loss, train_loader, validation_loader):
    best_validation = validation_min
    for epoch in range(start_epoch, epochs):
        info_logger.info("Starting epoch: {}".format(epoch+1))

        train_loss = loss_pass(net, loss, train_loader, optimizer=optimizer, prints=True)

        #train_loss = loss_pass(net, loss, train_loader)
        validation_loss = loss_pass(net, loss, validation_loader)

        info_logger.info("Finished epoch {}".format(epoch+1))
        info_logger.info("Train loss = {:.12f}".format(train_loss))
        info_logger.info("Validation loss = {:.12f}".format(validation_loss))
        data_logger.info("{}, {:12f}, {:12f}".format(epoch, train_loss, validation_loss))

        if validation_loss < best_validation:
            info_logger.info("New best validation! Saving")
            torch.save(net.state_dict(), validation_net_path)
            best_validation = validation_loss

        torch.save(net.state_dict(), final_net_path)

def loss_pass(net, loss, data_loader, optimizer=None, prints=False):
    batches = len(data_loader)
    if batches == 0:
        return 0

    if prints:
        print_every = 5
        start_time = time.time()
    
    running_loss = 0.0
    total_loss = 0
    for i, data in enumerate(data_loader, 0):
        airfoils, pressures = data
        airfoils, pressures = Variable(airfoils), Variable(pressures)
        
        if optimizer:
            optimizer.zero_grad()

        pressures_pred = net(airfoils)
        loss_size = loss(pressures_pred, pressures)

        if optimizer:
            loss_size.backward()
            optimizer.step()

        total_loss += loss_size.item()

        if prints:
            running_loss += loss_size.item()

            #Print every nth batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                info_logger.info("Loss: {:.12f} took: {:.2f}s".format(
                        running_loss / (print_every * data_loader.batch_size),
                        time.time() - start_time))
                running_loss = 0.0
                start_time = time.time()

    return total_loss / len(data_loader.dataset)

if __name__ == "__main__":main()

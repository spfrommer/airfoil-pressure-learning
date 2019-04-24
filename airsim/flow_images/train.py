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
from airflow_unet import Airflow_Unet256
import dataset
import pdb

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import airsim.dirs as dirs
from airsim.io_utils import empty_dir

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

training_plots_i = np.array([1, 2, 3, 4, 5])
valid_plots_i = np.array([1, 2, 3, 4, 5])

resume = False

sdf_samples = False
validation_net_path = dirs.out_path('training', 'validation_net.pth')
final_net_path = dirs.out_path('training', 'final_net.pth')

epochs = 3
num_workers = 0
batch_size = 3
learning_rate = 0.001 * (batch_size / 64.0)
#learning_rate_mul = 0.8
#learning_rate_mul_interval = 2000 # Number of descents per lr rescale

append = False
load_net_path = None
start_epoch = 0
validation_min = float('inf')


#global arrays for plotting
epoch_idx = []
train_loss_idx = []
valid_loss_idx = []

if resume:
    append = True
    load_net_path = final_net_path
    log_path = dirs.out_path('training', 'data.log') 
    array = np.loadtxt(open(log_path, "rb"), delimiter=",", skiprows=0)
    start_epoch = np.size(array, 0)
    validation_min = np.min(array[:,2])
else:
    empty_dir(dirs.out_path('training'))

writer = SummaryWriter(dirs.out_path('training', 'runs'))

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
    #net = GuoCNN(sdf=sdf_samples).to(device)
    net = Airflow_Unet256((1, 256, 256), sdf_samples).to(device)
    
    if load_net_path:
        net.load_state_dict(torch.load(load_net_path))
    loss = losses.foil_mse_loss
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    #optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate,
    #        eps=1e-4, weight_decay=0, momentum=0.9)

    training_start_time = time.time()
    try:
        train(net, optimizer, loss, train_loader, validation_loader)
    except Exception:
        info_logger.exception("Error in training")

    info_logger.info("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    test_loss = loss_pass(net, loss, test_loader, -1)
    info_logger.info("Test loss = {:.12f}".format(test_loss))
    writer.add_scalar('Test Loss', test_loss)

def setup_multiprocessing():
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

info_logger, data_logger = logs.get_loggers()

def train(net, optimizer, loss, train_loader, validation_loader):
    best_validation = validation_min
    for epoch in range(start_epoch, epochs):
        info_logger.info("Starting epoch: {}".format(epoch+1))
        train_loss = loss_pass(net, loss, train_loader, epoch+1, optimizer=optimizer, log=True)

        #train_loss = loss_pass(net, loss, train_loader)
        validation_loss = loss_pass(net, loss, validation_loader, epoch+1)
        log_epoch_loss(epoch + 1, train_loss, validation_loss)

        info_logger.info("Finished epoch {}".format(epoch+1))
        info_logger.info("Train loss = {:.12f}".format(train_loss))
        info_logger.info("Validation loss = {:.12f}".format(validation_loss))
        data_logger.info("{}, {:12f}, {:12f}".format(epoch, train_loss, validation_loss))

        if validation_loss < best_validation:
            info_logger.info("New best validation! Saving")
            torch.save(net.state_dict(), validation_net_path)
            best_validation = validation_loss
            writer.add_scalar('Best Validation Loss', best_validation)
        torch.save(net.state_dict(), final_net_path)

def log_epoch_loss(epoch_num, train_loss, validation_loss):
    fig = plt.figure(figsize=(8, 8))
    epoch_idx.append(epoch_num)
    train_loss_idx.append(train_loss)
    valid_loss_idx.append(validation_loss)
    ax = []
    ax.append(fig.add_subplot(1, 1, 1))
    plt.title("MSE Train / Validation Loss across each Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.plot(epoch_idx, train_loss_idx, color='#85144b', label='train_loss', lw=3)
    plt.plot(epoch_idx, valid_loss_idx, color='#FF851B', label='valid_loss', lw=3)
    plt.grid(True)
    plt.legend()
    label = 'Loss Graph'
    writer.add_figure(label, fig)

#Change this method to show the same plot
def log_batch_output(x, y, y_hat, sample_id, epoch, train=False, cmap='coolwarm'):        
    if(len(sample_id > 0)):
        for i in range(x.shape[0]):
            if ((train and np.isin(sample_id[i], training_plots_i)) or (not train and np.isin(sample_id[i], valid_plots_i))):
                fig = plt.figure(figsize=(12, 12))
                fig.suptitle('Input Image, Ground_Truth, Prediction, Absolute Difference- Epoch {}'.format(epoch))
                ax = []
                ax.append(fig.add_subplot(2, 2, 1))
                plt.imshow(x[i, :, :], cmap=cmap)
                plt.colorbar(fraction=0.046, pad=0.04)
                ax.append(fig.add_subplot(2, 2, 2))
                plt.imshow(y[i, :, :], cmap=cmap)
                plt.colorbar(fraction=0.046, pad=0.04)
                ax.append(fig.add_subplot(2, 2, 3))
                plt.imshow(y_hat[i, :, :], cmap=cmap)
                plt.colorbar(fraction=0.046, pad=0.04)
                ax.append(fig.add_subplot(2, 2, 4))
                plt.imshow(y_hat[i, :, :] - y[i, :, :], cmap=cmap)
                plt.colorbar(fraction=0.046, pad=0.04)
                if(train):
                    label = 'TRAIN:Plots Id : {}'.format(sample_id[i])
                else:
                    label = 'VALID:Plots Id : {}'.format(sample_id[i])
                writer.add_figure(label, fig)


def find_matching_ids(batch_sample_ids, target_sample_ids):
    return np.in1d(batch_sample_ids, target_sample_ids)

def loss_pass(net, loss, data_loader, epoch_num, optimizer=None, log=False):
    batches = len(data_loader)
    if batches == 0:
        return 0

    if log:
        print_every = len(data_loader) - 1
        start_time = time.time()
    
    if (optimizer == None):
        torch.autograd.set_grad_enabled(False)
    else:
        torch.autograd.set_grad_enabled(True)

    running_loss = 0.0
    total_loss = 0
    for i, data in enumerate(data_loader, 0):
        airfoils, pressures, sample_ids = data
        airfoils, pressures = Variable(airfoils), Variable(pressures)
        
        if optimizer:
            optimizer.zero_grad()

        pressures_pred = net(airfoils)
        loss_size = loss(pressures_pred, pressures)
        
        if optimizer:
            loss_size.backward()
            optimizer.step()
            #matching_ids = find_matching_ids(sample_ids.cpu(), training_plots_i)
            #matching_ids = np.where(matching_ids)[0]
            airfoils = ((torch.squeeze(airfoils.cpu(), dim = 1)).numpy())
            pressures = ((torch.squeeze(pressures.cpu(), dim=1)).numpy())
            pressures_pred  = (pressures_pred.detach().cpu().numpy())
            log_batch_output(airfoils, pressures, pressures_pred, sample_ids, epoch_num, train=True)

        else:
            #matching_ids = find_matching_ids(sample_ids, valid_plots_i)
            airfoils = ((torch.squeeze(airfoils.cpu(), dim = 1)).numpy())
            pressures = ((torch.squeeze(pressures.cpu(), dim=1)).numpy())
            pressures_pred  = (pressures_pred.detach().cpu().numpy())
            log_batch_output(airfoils, pressures, pressures_pred, sample_ids, epoch_num, train=False)
            

        total_loss += loss_size.item()
        if log:
            running_loss += loss_size.item()

            #Print every nth batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                timetaken = time.time() - start_time
                info_logger.info("Loss: {:.12f} took: {:.2f}s".format(
                        running_loss / (print_every * data_loader.batch_size),
                        timetaken))
                writer.add_scalar('Time Taken', timetaken)
                running_loss = 0.0
                start_time = time.time()
    torch.autograd.set_grad_enabled(True)
    return total_loss / len(data_loader.dataset)

if __name__ == "__main__":main()

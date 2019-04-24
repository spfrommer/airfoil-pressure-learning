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
from torch.optim.lr_scheduler import StepLR

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

resume = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

training_plots_i = np.array([1, 2, 3, 4, 5])
valid_plots_i = np.array([1, 2, 3, 4, 5])

sdf_samples = False
validation_net_path = dirs.out_path('training', 'validation_net.pth')
final_net_path = dirs.out_path('training', 'final_net.pth')

epochs = 500
num_workers = 1
batch_size = 64
learning_rate_base = 0.0004 * (batch_size / 64.0)

append = False
load_net_path = None
start_epoch = 0
validation_min = float('inf')

if resume:
    append = True
    load_net_path = final_net_path
    log_path = dirs.out_path('training', 'data.log') 
    array = np.loadtxt(open(log_path, "rb"), delimiter=",", skiprows=0)
    start_epoch = np.size(array, 0)
    validation_min = np.min(array[:,2])

writer = SummaryWriter(dirs.out_path('training', 'runs'))

def main():
    if num_workers > 0:
        setup_multiprocessing()
        
    if not resume:
        empty_dir(dirs.out_path('training'))
        empty_dir(dirs.out_path('training', 'runs'))

    info_logger, data_logger = logs.create_loggers(training=True, append=append)

    if resume:
        info_logger.info("Resuming training")
        info_logger.info("Start epoch: {}".format(start_epoch))
        info_logger.info("Best validation error: {}".format(validation_min))

    (train_dataset, train_loader, validation_dataset, validation_loader,
            test_dataset, test_loader) = dataset.load_data(sdf_samples, device, batch_size, num_workers)
    net = GuoCNN(sdf=sdf_samples).to(device)
    #net = Airflow_Unet256((1, 256, 256), sdf_samples).to(device)
    
    if load_net_path:
        net.load_state_dict(torch.load(load_net_path))
    loss = losses.foil_mse_loss
    #loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate_base)
    #optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate_base,
    #        eps=1e-4, weight_decay=0, momentum=0.9)

    training_start_time = time.time()
    try:
        train(net, optimizer, loss, train_loader, validation_loader)
    except Exception:
        info_logger.exception("Error in training")

    info_logger.info("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    test_loss, test_loss_percent = loss_pass(net, loss, test_loader, -1)
    info_logger.info("Test loss = {:.12f}".format(test_loss))
    info_logger.info("Test loss percent = {:.12f}".format(test_loss_percent))
    writer.add_scalar('Test Loss', test_loss)

def setup_multiprocessing():
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

info_logger, data_logger = logs.get_loggers()

def train(net, optimizer, loss, train_loader, validation_loader):
    elapsed_epochs = []
    train_losses = []
    validation_losses = []
    train_losses_percent = []
    validation_losses_percent = []

    best_validation = validation_min
    
    scheduler = StepLR(optimizer, step_size=30, gamma=0.3)
    
    for epoch in range(start_epoch, epochs):
        info_logger.info("Starting epoch: {}".format(epoch+1))
        train_loss, train_loss_percent = loss_pass(net, loss, train_loader, epoch+1, optimizer=optimizer, log=True)
        validation_loss, validation_loss_percent = loss_pass(net, loss, validation_loader, epoch+1)
        
        scheduler.step()

        elapsed_epochs.append(epoch+1)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        train_losses_percent.append(train_loss_percent * 100)
        validation_losses_percent.append(validation_loss_percent * 100)

        log_epoch_loss(elapsed_epochs, train_losses,
                       validation_losses, 'MSE Loss')
        log_epoch_loss(elapsed_epochs, train_losses_percent,
                       validation_losses_percent, 'Median Percent Error')

        info_logger.info("Finished epoch {}".format(epoch+1))
        info_logger.info("Train loss = {:.12f}".format(train_loss))
        info_logger.info("Train loss percent = {:.12f}".format(train_loss_percent))
        info_logger.info("Validation loss = {:.12f}".format(validation_loss))
        info_logger.info("Validation loss percent = {:.12f}".format(validation_loss_percent))
        data_logger.info("{}, {:12f}, {:12f}, {:12f}, {:12f}".format(epoch,
            train_loss, validation_loss, train_loss_percent, validation_loss_percent))

        if validation_loss < best_validation:
            info_logger.info("New best validation! Saving")
            torch.save(net.state_dict(), validation_net_path)
            best_validation = validation_loss
            writer.add_scalar('Best Validation Loss', best_validation)

        torch.save(net.state_dict(), final_net_path)

def log_epoch_loss(epochs, train_losses, valid_losses, label):
    info_logger.info('Logging epoch loss image')
    fig = plt.figure(figsize=(8, 8))
    ax = []
    ax.append(fig.add_subplot(1, 1, 1))
    plt.title("Train & Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel(label)
    plt.plot(epochs, train_losses, color='#85144b', label='train_loss', lw=3)
    plt.plot(epochs, valid_losses, color='#FF851B', label='valid_loss', lw=3)
    plt.grid(True)
    plt.legend()
    writer.add_figure(label, fig)

#Change this method to show the same plot
def log_batch_output(x, y, y_hat, sample_id, epoch, train=False, cmap='coolwarm'):        
    info_logger.info('Logging batch image')
    x = torch.squeeze(x.cpu(), dim=1).numpy()
    y = torch.squeeze(y.cpu(), dim=1).numpy()
    y_hat  = torch.squeeze(y_hat.detach().cpu(), dim=1).numpy()

    if (len(sample_id) > 0):
        for i in range(x.shape[0]):
            if ((train and np.isin(sample_id[i], training_plots_i)) or (not train and np.isin(sample_id[i], valid_plots_i))):
                fig = plt.figure(figsize=(12, 12))
                fig.suptitle('Input Image, Ground_Truth, Prediction, Absolute Difference | Epoch {}'.format(epoch))
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
        print_every = 5
        start_time = time.time()
    
    render_every = 10

    running_loss = 0.0
    total_loss = 0
    total_percentage_loss = 0
    
    for i, data in enumerate(data_loader, 0):
        airfoils, pressures, sample_ids = data
        airfoils, pressures = Variable(airfoils), Variable(pressures)
        
        if optimizer:
            optimizer.zero_grad()

        pressures_pred = net(airfoils)
        loss_size = loss(pressures_pred, pressures)
        percentage_loss_size = losses.median_percentage_loss(pressures_pred, pressures)
        
        if optimizer:
            loss_size.backward()
            optimizer.step()
        
        if epoch_num % render_every == 0:
            log_batch_output(airfoils, pressures, pressures_pred, sample_ids, epoch_num, train=(optimizer is not None))
            
        total_loss += loss_size.item()
        total_percentage_loss = percentage_loss_size.item()
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

    return total_loss / len(data_loader.dataset), total_percentage_loss / len(data_loader.dataset)

if __name__ == "__main__":main()

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
from guocnn import GuoCNN
from dataset import ProcessedAirfoilDataset

import airsim.dirs as dirs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sdf_samples = False
load_net = True
append = True
net_path = dirs.out_path('trained', 'net.pth')
start_epoch = 83
epochs = 300
batch_size = 64
learning_rate = 0.0001

def main():
    setup_multiprocessing()
    info_logger, data_logger = logs.create_training_loggers(append=append)
    train_dataset, train_loader, test_dataset, test_loader = init_data()
    net = GuoCNN().to(device)
    if load_net:
        net.load_state_dict(torch.load(net_path))
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    training_start_time = time.time()
    try:
        train(net, optimizer, loss, train_loader, test_loader)
    except Exception:
        info_logger.exception("Error in training")
    info_logger.info("Training finished, took {:.2f}s".format(time.time() - training_start_time))

def setup_multiprocessing():
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    #try:
        #mp.set_start_method('spawn')
    #except RuntimeError:
        #pass

info_logger, data_logger = logs.get_training_loggers()

def init_data():
    train_dataset = ProcessedAirfoilDataset(
            dirs.out_path('processed', 'train'), sdf_samples, device, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    test_dataset = ProcessedAirfoilDataset(
            dirs.out_path('processed', 'test'), sdf_samples, device, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=2)
    info_logger.info("Training dataset size: {}".format(len(train_dataset)))
    info_logger.info("Testing dataset size: {}".format(len(test_dataset)))
    return train_dataset, train_loader, test_dataset, test_loader

def train(net, optimizer, loss, train_loader, test_loader):
    batches = len(train_loader)
    info_logger.info("Got {} batches".format(batches))
    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        print_every = 5
        start_time = time.time()
        total_train_loss = 0
        
        # Go through entire training dataset
        for i, data in enumerate(train_loader, 0):
            airfoils, pressures = data
            airfoils, pressures = Variable(airfoils), Variable(pressures)
            
            optimizer.zero_grad()
            
            pressures_pred = net(airfoils)
            loss_size = loss(pressures_pred, pressures)
            loss_size.backward()
            optimizer.step()
            
            running_loss += loss_size.item()
            total_train_loss += loss_size.item()
            
            #Print every nth batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                info_logger.info("Epoch {}, {:d}% \t train loss: {:.12f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / batches), running_loss / (print_every * batch_size),
                        time.time() - start_time))
                running_loss = 0.0
                start_time = time.time()
            
        # At the end of the epoch, do a pass on the test set
        total_test_loss = 0
        for airfoils, pressures in test_loader:
            airfoils, pressures = Variable(airfoils), Variable(pressures)

            pressures_pred = net(airfoils)
            loss_size = loss(pressures_pred, pressures)
            total_test_loss += loss_size.item()
           
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_test_loss = total_test_loss / len(test_loader.dataset)
 
        info_logger.info("Finished epoch {}".format(epoch+1))
        info_logger.info("Train loss = {:.12f}".format(avg_train_loss))
        info_logger.info("Test loss = {:.12f}".format(avg_test_loss))
        data_logger.info("{}, {}, {}".format(epoch, avg_train_loss, avg_test_loss))
        torch.save(net.state_dict(), net_path)

if __name__ == "__main__":main()

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

sdf_samples = True
test_trained_net = True
load_net = True
net_path = dirs.out_path('trained', 'net.pth')
epochs = 0
batch_size = 64
learning_rate = 0.0001

def main():
    #setup_multiprocessing()
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

    if test_trained_net:
        test_net(net, test_dataset[0])

def setup_multiprocessing():
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    #try:
        #mp.set_start_method('spawn')
    #except RuntimeError:
        #pass

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

def train(net, optimizer, loss, train_loader, test_loader):
    batches = len(train_loader)
    info_logger.info("Got {} batches".format(batches))
    for epoch in range(epochs):
        running_loss = 0.0
        print_every = 1
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
                info_logger.info("Epoch {}, {:d}% \t train loss: {:.2f} took: {:.2f}s".format(
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
           
        avg_train_loss = total_train_loss / (len(train_loader) * batch_size)
        avg_test_loss = total_test_loss / (len(test_loader) * batch_size)
 
        info_logger.info("Finished epoch {}".format(epoch+1))
        info_logger.info("Train loss = {:.2f}".format(avg_train_loss))
        info_logger.info("Test loss = {:.2f}".format(avg_test_loss))
        data_logger.info("{}, {}, {}".format(epoch, avg_train_loss, avg_test_loss))
        torch.save(net.state_dict(), net_path)


def test_net(net, sample):
    airfoil, pressure = sample
    airfoil, pressure = Variable(airfoil), Variable(pressure)
    airfoil = airfoil.view(1, 1, 256, 256)
    pressure = pressure.view(1, 1, 256, 256)
    pressure_pred = net(airfoil)
    pressure_error = pressure_pred - pressure

    info_logger.info(airfoil)
    utils.save_image(airfoil, dirs.out_path('trained', 'airfoil.png'))

    info_logger.info(pressure)
    pressure = (pressure / 400) + 0.5
    utils.save_image(pressure, dirs.out_path('trained', 'pressure.png'))

    info_logger.info(pressure_pred)
    pressure_pred = (pressure_pred / 200) + 0.5
    utils.save_image(pressure_pred, dirs.out_path('trained', 'pressure_pred.png'))

    info_logger.info(pressure_error)
    pressure_error = (pressure_error / 200) + 0.5
    utils.save_image(pressure_error, dirs.out_path('trained', 'pressure_error.png'))

if __name__ == "__main__":main()

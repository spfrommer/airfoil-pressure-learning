from tensorboardX import SummaryWriter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from airsim import dirs

font_small = 18
font_medium = 20
font_big = 22

plt.rc('font', size=font_small)
plt.rc('axes', titlesize=font_small)
plt.rc('axes', labelsize=font_medium)
plt.rc('xtick', labelsize=font_small)
plt.rc('ytick', labelsize=font_small)
plt.rc('legend', fontsize=font_small)
plt.rc('figure', titlesize=font_big)

writer = None
training_plots_i = None
valid_plots_i = None

def init():
    global writer, training_plots_i, valid_plots_i
    writer = SummaryWriter(dirs.out_path('training', 'runs'))
    training_plots_i = np.array([1, 2, 3, 4, 5])
    valid_plots_i = np.array([1, 2, 3, 4, 5])

def log_batch_output(x, y, y_hat, sample_id, epoch, sdf_samples=False, train=False):
    cmap=matplotlib.cm.coolwarm
    cmap.set_bad(color='black')
    loaded = False

    if (len(sample_id) > 0):
        for i in range(x.shape[0]):
            if ((train and np.isin(sample_id[i], training_plots_i)) or (not train and np.isin(sample_id[i], valid_plots_i))):
                if not loaded:
                    if sdf_samples:
                        x = torch.squeeze(x.cpu(), dim=1).numpy()
                    else:
                        x = torch.squeeze(zeros_to_nan(x.cpu()), dim=1).numpy()

                    y = torch.squeeze(zeros_to_nan(y.cpu()), dim=1).numpy()
                    y_hat  = torch.squeeze(zeros_to_nan(y_hat.detach().cpu()), dim=1).numpy()
                    loaded = True

                fig = plt.figure(figsize=(12, 12))
                fig.suptitle('Pressure Fields | Epoch {}'.format(epoch))
                ax = []
                ax.append(fig.add_subplot(2, 2, 1))
                plt.title('Input Image')
                render_image(x[i, :, :], cmap, center=False)
                ax.append(fig.add_subplot(2, 2, 2))
                plt.title('Ground Truth')
                render_image(y[i, :, :], cmap)
                ax.append(fig.add_subplot(2, 2, 3))
                plt.title('Prediction', y=-0.1)
                render_image(y_hat[i, :, :], cmap)
                ax.append(fig.add_subplot(2, 2, 4))
                plt.title('Error', y=-0.1)
                render_image(y_hat[i, :, :] - y[i, :, :], cmap)

                for axis in ax:
                    axis.set_xticks([])
                    axis.set_yticks([])

                plt.tight_layout()
                plt.subplots_adjust(hspace=-0.1, top=0.94)

                if train:
                    label = 'TRAIN:Plots Id : {}'.format(sample_id[i])
                else:
                    label = 'VALID:Plots Id : {}'.format(sample_id[i])
                writer.add_figure(label, fig)

def log_epoch_loss(epochs, train_losses, valid_losses, label):
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

def render_image(image, cmap, center=True):
    max_deviation = max(np.nanmin(image), np.nanmax(image), key=abs) 
    if center:
        plt.imshow(image, vmin=-max_deviation,
                   vmax=max_deviation, cmap=cmap)
    else:
        plt.imshow(image, cmap=cmap)
    plt.colorbar(fraction=0.046, pad=0.04)

def zeros_to_nan(tensor):
    return torch.where(tensor == 0, torch.ones(tensor.size()) * float('nan'), tensor)

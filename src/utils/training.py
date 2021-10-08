from IPython.display import clear_output

import time
import copy
import os
import json

import numpy as np
import torch
from .visual import multiplot

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(
                f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


def fit(model, data_loaders, dataset_sizes, criterion,
        optimizer, early_stopping, scheduler=None, clear= 'notebook',
        num_epochs=100, device="cuda", plot=True, metrics_path = None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    metrics = []
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []
    epochs = []

    try:
        for epoch in np.arange(num_epochs) + 1:
            epochs.append(epoch)
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in data_loaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = (running_corrects.double()).item() / \
                    dataset_sizes[phase]
                if phase == 'train':
                    if scheduler is not None:
                        scheduler.step()
                    train_loss.append(epoch_loss)
                    train_acc.append(epoch_acc)
                else:
                    val_loss.append(epoch_loss)
                    val_acc.append(epoch_acc)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                    early_stopping(epoch_loss)
            if clear == 'notebook':
                clear_output(wait=True)
            elif clear == 'terminal':
                os.system('cls' if os.name == 'nt' else 'clear')
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-' * 10)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                'Train', train_loss[-1], train_acc[-1]))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                'Val', val_loss[-1], val_acc[-1]))
            print()
            if plot == True:
                metrics = np.array([epochs, train_loss, train_acc,
                                    val_loss, val_acc])
                plot_metrics(metrics, metrics_path)
            if early_stopping.early_stop:
                break
    except KeyboardInterrupt:
        print("Training interrupted.")
        pass
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    model.aux_dict['train_iters'] += len(train_loss)

    return metrics

def test_model(model, test_loaders, device="cuda"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loaders:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = (100 * correct / total)
    print('Accuracy of the network on test images: %.2f %%' %
          (accuracy))
    model.aux_dict['test_acc'] = accuracy
    return accuracy


def load_model(model, path, optim=False, get_dataloaders=False,
               get_train_metrics=False, device="cuda"):
    output = []
    model.load_state_dict(torch.load(
        path + '_parameters.pt', map_location=device))
    with open(path + '_aux_dict.json') as json_file:
        model.aux_dict = json.load(json_file)
    if optim:
        optim.load_state_dict(torch.load(
            path + '_optim.pt', map_location=device))
    if get_dataloaders:
        data_loaders = torch.load(
            path + '_dataloaders.pt', map_location=device)
        output.append(data_loaders)
    if get_train_metrics:
        metrics = np.load(path + '_train_metrics.npy')
        output.append(metrics)
    return output


def save_model(model, path, optim=None,dataloaders=None, train_metrics=None):

    torch.save(model.state_dict(), path + '_parameters.pt')
    with open(path + '_aux_dict.json', 'w') as json_file:
        json.dump(model.aux_dict, json_file)
    if optim is not None:
        torch.save(optim.state_dict(), path + '_optim.pt')
    if dataloaders is not None:
        torch.torch.save(dataloaders, path + '_dataloaders.pt')
    if train_metrics is not None:
        np.save(path + '_train_metrics.npy', train_metrics)


def plot_metrics(metrics, save_path=None):
    epochs = metrics[0]
    remaining_metrics = metrics[1:]
    systems = np.array([[epochs, metric] for metric in remaining_metrics])
    labels = ['Training', 'Validation']
    multiplot(systems[[0, 2]], 'Epoch', 'Loss',
              labels, save_path + '_loss_comparison.pdf')
    multiplot(systems[[1, 3]], 'Epoch', 'Accuracy',
              labels, save_path + '_accuracy_comparison.pdf')

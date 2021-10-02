from IPython.display import clear_output

import pathlib as Path
import time
import copy

import numpy as np
import torch
from src.utils.visual import multiplot

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
        optimizer, early_stopping, scheduler=None,
        num_epochs=100, device="cuda:0", plot=False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss, train_acc = [], []
    val_loss, val_acc = [], []
    epochs = []

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
                if scheduler != None:
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
        clear_output(wait=True)
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'Train', train_loss[-1], train_acc[-1]))
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'Val', val_loss[-1], val_acc[-1]))
        print()
        if plot == True:
            current_accs = np.array(train_acc[:epoch], val_acc[:epoch])
            systems = np.array([[epochs, current_acc]
                               for current_acc in current_accs])
            plot_labels = ['Training', 'Validation']
            multiplot(systems, 'Epoch', 'Accuracy', plot_labels)
        if early_stopping.early_stop:
            break
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    metrics = np.array([epochs, train_loss, train_acc,
                        val_loss, val_acc]).T
    return metrics

def test_model(model, test_loaders, device="cuda:0"):
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
    return accuracy


def load_model(model, name, optim=False, get_dataloaders=False,
               get_train_metrics=False, get_test_scores=False,
               device="cuda:0"):
    output = []
    path = 'models/' + name
    model.load_state_dict(torch.load(
        path + '_parameters.pt', map_location=device))
    if optim:
        optim.load_state_dict(torch.load(
            path + '_optim.pt', map_location=device))
    if get_dataloaders:
        data_loaders = torch.load(
            path + '_dataloaders.pt', map_location=device)
        output.append(data_loaders)
    if get_train_metrics:
        metrics = np.load(path + '_metrics.npy')
        output.append(metrics)
    if get_test_scores:
        test_scores = np.load(path + '_test_scores.npy')
        output.append(test_scores)
    return output


def save_model(model, name='model_0', optim=None,
               dataloaders=None, train_metrics=None,
               test_scores=None):
    Path("models").mkdir(parents=True, exist_ok=True)
    path = 'models/' + name
    torch.save(model.state_dict(), path + '_parameters.pt')
    if optim != None:
        torch.save(optim.state_dict(), path + '_optim.pt')
    if dataloaders != None:
        torch.torch.save(dataloaders, path + '_dataloaders.pt')
    if train_metrics != None:
        np.save(path + '_train_metrics.npy', train_metrics)
    if test_scores != None:
        np.save(path + '_test_scores.npy', test_scores)


def plot_metrics(metrics, name='model_0'):
    epochs = metrics[:, 0]
    remaining_metrics = metrics[:, 1:]
    systems = np.array([[epochs, metric] for metric in remaining_metrics.T])
    labels = ['Training', 'Validation']
    multiplot(systems[[0, 2]], 'Epoch', 'Loss',
              labels, name + '_loss_comparison.pdf')
    multiplot(systems[[1, 3]], 'Epoch', 'Accuracy',
              labels, name + '_accuracy_comparison.pdf')

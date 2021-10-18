from IPython.display import clear_output

import time
import copy
import os
import json

import numpy as np
import torch

from .custom_types import ModelConfig, TrainingConfig, DatasetConfig, PlotConfig

from .visual import plot_metrics
from .config import RNG_SEED, DEFAULT_MODEL_PATH, DEFAULT_METRICS_PATH, save_training_metadata
from ..models import save_model

#IMPLEMENT smarter early stopping that calculates graph trend based on last N values
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0, min_epochs=0):
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
        self.epochs = 0
        self.min_epochs = min_epochs

    def __call__(self, val_loss):
        self.epochs += 1
        if self.best_loss == None:
            self.best_loss = val_loss
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            if self.epochs > self.min_epochs:
                self.counter += 1
                print(
                    f"INFO: Early stopping counter {self.counter} of {self.patience}")
                if self.counter >= self.patience:
                    print('INFO: I have no time for your silly games. Stopping early.')
                    self.early_stop = True
    def __repr__(self):
        args = 'patience = {}, min_delta = {}, min_epochs = {}'.format(self.patience, self.min_delta, self.min_epochs)
        return self.__class__.__name__ + '({})'.format(args)


#IMPLEMENT rolling average
# TODO incorcorate model into model_config and data_loaders, dataset_sizes into dataset_config or a similar dicts
def fit(model, data_loaders, dataset_sizes,
        model_config: ModelConfig, training_config: TrainingConfig, dataset_config: DatasetConfig, 
        plot_config: PlotConfig,
        clear='terminal', plot=False, save_interval=25):
    assert model_config is not None

    device = str(model_config['device'])
    num_epochs = training_config['train_info']['num_epochs']
    best_model_wts = copy.deepcopy(model.state_dict())
    best_model_epochs = 0
    best_acc = 0.0
    metrics = []
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []
    train_loss_avg, train_acc_avg = [], []
    val_loss_avg, val_acc_avg = [], []
    epochs = []
    
    # TODO since we already have criterion in training_config why not just add cross_entropy as default there?
    criterion = torch.nn.CrossEntropyLoss()

    optim_args = training_config['optim_args']
    # TODO the same could be said about the optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=optim_args['lr'], eps=optim_args['eps'])

    lr_decay = training_config['train_info']['lr_decay']
    # TODO also consider the same for scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    es_args = training_config['early_stopping_args']
    # TODO The same
    early_stopping = EarlyStopping(min_epochs = es_args['min_epochs'], 
                        patience=es_args['patience'], min_delta=es_args['min_delta'])
    # TODO add these as parameters in eg training_config
    metrics_path = DEFAULT_METRICS_PATH
    model_path   = DEFAULT_MODEL_PATH
    
    training_config['criterion'] = criterion
    training_config['optim'] = optimizer
    training_config['early_stopping'] = early_stopping
    training_config['scheduler'] = scheduler

    since = time.time()
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
                    
                        # TODO make this part of the code a subprocedure that can switched in and out, eg give it as a parameter
                        # So that we can later apply training to models with a more complicated output format as well as using different
                        # metrics
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                # TODO if we implement the above TODO then this code might also need to be changed depending on which metrics and model are used
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = (running_corrects.double()).item() / dataset_sizes[phase]


                if phase == 'train':
                    train_loss.append(epoch_loss)
                    train_acc.append(epoch_acc)

                    #rolling average
                    train_loss_avg.append(np.mean(train_loss[-plot_config['rolling_avg_window']:len(train_loss)]))
                    train_acc_avg.append(np.mean(train_acc[-plot_config['rolling_avg_window']:len(train_acc)]))
                else:
                    val_loss.append(epoch_loss)
                    val_acc.append(epoch_acc)

                    #rolling average
                    val_loss_avg.append(np.mean(val_loss[-plot_config['rolling_avg_window']:len(val_loss)]))
                    val_acc_avg.append(np.mean(val_acc[-plot_config['rolling_avg_window']:len(val_acc)]))
                    
                    # TODO consider choosing final model based on loss and not accuracy

                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        best_model_epochs = int(epoch)
                    early_stopping(epoch_loss)
            # TODO this clear == notebook might need to be changed if we are using widgets
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
            metrics = np.array([epochs, train_loss, train_acc,
                                    val_loss, val_acc])
            if plot == True:
                # QUESTION why not just save all metrics in "metrics" to begin with?
                plotting_metrics = np.array([[train_loss, train_loss_avg,
                                            val_loss, val_loss_avg],
                                            [train_acc, train_acc_avg,
                                            val_acc, val_acc_avg]])
                plot_metrics(plot_config, np.array(epochs), plotting_metrics, metrics_path+model_config['model_name'])
            scheduler.step()
            if epoch % save_interval == 0:
                temp_state_dict = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_model_wts)
                # TODO maybe just pass the model state dict to save rather than the model itself?
                training_config['train_info']['trained_epochs'] = best_model_epochs
                save_model(model, model_path+model_config['model_name'], optim=None,dataloaders=data_loaders, train_metrics=metrics)
                save_training_metadata(model_path+model_config['model_name'], model_config, dataset_config, training_config)
                model.load_state_dict(temp_state_dict)
            if early_stopping.early_stop:
                training_config['train_info']['stopped_early'] = True
                break
    except KeyboardInterrupt:
        print("Training interrupted.")
        pass
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # TODO maybe also print best loss
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    # QUESTION any way of avoiding the code below?
    training_config['train_info']['trained_epochs'] = best_model_epochs
    save_model(model, model_path+model_config['model_name'], optim=None,dataloaders=data_loaders, train_metrics=metrics)
    save_training_metadata(model_path+model_config['model_name'], model_config, dataset_config, training_config)

    return metrics

def test_model(model, test_loaders, training_config: TrainingConfig, device="cuda"):
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
    training_config['train_info']['test_acc'] = accuracy
    return accuracy



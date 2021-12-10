from IPython.display import clear_output

import time
import copy
import os
import typing as t
import numpy.typing as npt

import numpy as np
import torch
from torch.utils.data import DataLoader

from .custom_types import ModelConfig, TrainingConfig, DatasetConfig, PlotConfig
from .config import DEFAULT_METRICS_PATH, DEFAULT_MODEL_PATH, save, DEVICE
from .visual import plot_metrics

#IMPLEMENT smarter early stopping that calculates graph trend based on last N values
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience : int = 5, min_delta : float =0, min_epochs : int=0) ->None:
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

    def __call__(self, val_loss : float) -> None:
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
    def __repr__(self) -> str:
        args = 'patience = {}, min_delta = {}, min_epochs = {}'.format(self.patience, self.min_delta, self.min_epochs)
        return self.__class__.__name__ + '({})'.format(args) # we dont really need this repr method here in any case


def fit(model : torch.nn.Module, data_loaders : t.Dict[str, DataLoader], dataset_sizes : t.Dict[str, int],
        model_config: ModelConfig, training_config: TrainingConfig, dataset_config: DatasetConfig, 
        plot_config: PlotConfig, clear='terminal', plot: bool=False, save_interval: int=25, discount = 0.3) -> npt.NDArray:
    assert model_config is not None

    device = model_config['device']
    num_epochs = training_config['train_info']['num_epochs']

    best_model_wts = copy.deepcopy(model.state_dict())


    training_config['model_path'] = DEFAULT_MODEL_PATH
    training_config['metrics_path'] = DEFAULT_METRICS_PATH
    training_config['train_info']['best_model_val_loss'] = float("inf")


    metrics = []
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []
    train_loss_avg, train_acc_avg = [], []
    val_loss_avg, val_acc_avg = [], []
    epochs = []
    
    criterion = torch.nn.CrossEntropyLoss()
    optim_args = training_config['optim_args']
    optimizer = torch.optim.Adam(model.parameters(), lr=optim_args['lr'], eps=optim_args['eps'])

    lr_decay = training_config['train_info']['lr_decay']
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    es_args = training_config['early_stopping_args']
    early_stopping = EarlyStopping(min_epochs = es_args['min_epochs'], 
                        patience=es_args['patience'], min_delta=es_args['min_delta'])


    
    training_config['criterion'] = criterion
    training_config['optim'] = optimizer
    training_config['early_stopping'] = early_stopping
    training_config['scheduler'] = scheduler

    since = time.time()
    try:
        for epoch in np.arange(num_epochs) + 1:
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
                        if model_config['model_architecture'] == 'googlenet' and model.training:
                            outputs, aux_outputs_2, aux_outputs_1 = outputs
                            loss = criterion(outputs, labels) + discount * (criterion(aux_outputs_2, labels) + criterion(aux_outputs_1, labels) )
                        else:
                            loss = criterion(outputs, labels) 
                        _, preds = torch.max(outputs, 1)
                        

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
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
                    
                    if epoch_loss < training_config['train_info']['best_model_val_loss']:
                        training_config['train_info']['best_model_val_loss'] = epoch_loss
                        training_config['train_info']['best_model_val_acc'] = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        training_config['train_info']['best_model_epochs'] = int(epoch)


                    early_stopping(epoch_loss)
            epochs.append(epoch)
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
            metrics = np.array([[epochs, train_loss, train_loss_avg,
                                 val_loss, val_loss_avg],
                                [epochs, train_acc, train_acc_avg,
                                 val_acc, val_acc_avg]])
            if plot == True:
                plot_metrics(plot_config, metrics[0,0], metrics[:,1:,:], 
                            training_config['metrics_path'] +model_config['model_name'])
            scheduler.step()
            if epoch % save_interval == 0:
                save(training_config['model_path']+model_config['model_name'], model_config, #consider not saving while running but only when done.
                     dataset_config, training_config, best_model_wts, None, data_loaders, metrics, None)
            if early_stopping.early_stop:
                training_config['train_info']['stopped_early'] = True
                break
    except KeyboardInterrupt:
        print("Training interrupted.")
        pass
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best model val loss: {:4f}'.format(
        training_config['train_info']['best_model_val_loss']))
    print('Best model val Acc: {:4f}'.format(
        training_config['train_info']['best_model_val_acc']))
    
    metrics = np.array([[epochs, train_loss, train_loss_avg,
                         val_loss, val_loss_avg],
                        [epochs, train_acc, train_acc_avg,
                         val_acc, val_acc_avg]])
    # load best model weights
    model.load_state_dict(best_model_wts)
    save(training_config['model_path']+model_config['model_name'], model_config,
         dataset_config, training_config, best_model_wts, None, data_loaders, metrics, None) # this one is redundant. the only thing changed at this point is possible stopped_early set to True. 

    return metrics

def test_model(model :  torch.nn.Module, test_loader : DataLoader, 
               training_config: TrainingConfig, device : torch.device=DEVICE,
                mode : t.Literal['acc', 'loss'] = 'acc'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
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



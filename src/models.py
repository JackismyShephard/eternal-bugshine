import torch
import copy
import numpy as np
import json
import torch.nn as nn
from torchvision import models
import typing as t
import pandas as pd
from .utils.custom_types import *
from .utils.config import DEFAULT_METRICS_PATH, DEFAULT_MODEL_PATH, DEVICE
from .utils.transforms import string_to_class

##GENERAL COMMENTS
#TODO make general get_model function which encapsulates the whole pipeline of (possibly) loading model config
# downloading a corresponding torch model, updating parameters and dicts, training the model, testing the model
# possibly saving the model and finally hooking the model.



#TODO rename to download_model
def get_model(model_config: ModelConfig, dataset_config: DatasetConfig = None):
    name = model_config['model_architecture']
    pretrained = model_config['pretrained']
    device = model_config['device']
    if name == 'resnet18':
        model = models.resnet18(pretrained = pretrained)
    elif name == 'resnet34':
        model = models.resnet34(pretrained = pretrained)
    elif name == 'googlenet':
        model = models.googlenet(pretrained=pretrained)
    else:
        model = models.resnet50(pretrained=pretrained)
    
    if dataset_config is not None:
        num_classes = dataset_config['num_classes']
        num_fc = model.fc.in_features
        model.fc = nn.Linear(num_fc, num_classes)
    else:
        num_classes = model.fc.in_features
    model = model.to(device)
    # TODO just add the model module to model_config instead of adding aux_dict to model perhaps?
    # we already have train_iters (or equivalent) saved in training_config['training_info']
    # , 'name' and 'pretrained' saved in model_config so we just need to remember to save
    # test_acc to model_config also
    model.aux_dict = {'name': name, 'pretrained': pretrained, 
                      'num_classes': num_classes, 'train_iters': 0, 'test_acc': None}
    # and then we dont need to return anything here
    return model


#TODO this function is kind of redundant now and so perhaps should be moved to old
def load_model(model, path, optim=False, get_dataloaders=False,
               get_train_metrics=False, device=DEVICE):
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

# TODO consider giving just model name, model path prefix and device as parameters.
# Then we can first load in all the relevant dicts, then find and add 
# the relevant model module to the loaded model dict (along with relevant metadata)
# using get_model() and finally update the parameters of this model with the saved model
# parameters.
# TODO consider making loading of metrics and optimizer state optional 
# #or remove the corresponding options from save_model
def load_model_weights_and_metrics(model: torch.nn.Module, model_config: ModelConfig):
    device = model_config['device']
    path = DEFAULT_MODEL_PATH + model_config['model_name']
    model.load_state_dict(torch.load(
        path + '_parameters.pt', map_location=device))
    metrics = np.load(path + '_train_metrics.npy')
    with open(path + '_aux_dict.json') as json_file:
        old_config = json.load(json_file)
    string_augs = old_config['dataset_info']['data_augmentations']
    real_augs = [string_to_class(json.loads(x)) for x in string_augs]
    old_config['dataset_info']['data_augmentations'] = real_augs
    old_config['dataset_info']['mean'] = np.array(old_config['dataset_info']['mean'])
    old_config['dataset_info']['std'] = np.array(old_config['dataset_info']['std'])
    old_config['model_info']['device'] = torch.device(old_config['model_info']['device'])
    return metrics, old_config['model_info'], old_config['dataset_info'], old_config['train_info']

#TODO I think this model should also call save_training_metadata (or similar meta data saving functions)
#TODO Consider removing saving of dataloaders as we wont be using them. 
def save_model(model, path, optim=None,dataloaders=None, train_metrics=None):

    torch.save(model.state_dict(), path + '_parameters.pt')
    if optim is not None:
        torch.save(optim.state_dict(), path + '_optim.pt')
    if dataloaders is not None:
        torch.save(dataloaders, path + '_dataloaders.pt')
    if train_metrics is not None:
        np.save(path + '_train_metrics.npy', train_metrics)

#TODO We might consider removing this later (or moving into old)
class Exposed_model(torch.nn.Module):
    def __init__(self, model, flatten_layer):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.aux_dict = model.aux_dict
        self.flatten_layer = flatten_layer
        self.layers = {}
        for param in self.model.parameters():
            param.requires_grad = False
        for (name, layer) in self.model.named_children():
            if isinstance(layer, nn.Sequential):
                for (i, sub_layer) in enumerate(layer):
                    self.layers[(name + '_'+str(i))] = sub_layer
            else:
                self.layers[name] = layer

    def forward(self, x, out_info):
        out_activations = {}
        for (name, layer) in self.layers.items():
            if name == self.flatten_layer:
                x = x.view(x.size(0), -1)
            x = layer(x)
            for (out_name, out_idxs) in out_info:
                if name == out_name:
                    if out_idxs == None:
                        out_activations[name] = x
                    else:
                        out_activations[name + '_' +
                                        str(out_idxs)] = x[:, out_idxs]
        return out_activations

class HookedModel(torch.nn.Module):
    """Augments a pytorch model with the ability to return activations from specific modules in the model.
       Intended to be used in the following way: 'retval, activations = model(input, list_of_targets)'"""
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.aux_dict: t.Dict[str, t.Any] = model.aux_dict
        self._hooks: t.Dict[str, t.Any] = {}
        self._activations: t.Dict[str, torch.Tensor] = {}
    
    def show_modules(self):
        """Prints the named modules in the internal model"""
        all_modules = self.model.named_modules()
        start_layers = []
        hidden_layers = []
        end_layers = []
        
        at_end = False
        temp = []
        current_hidden = 1
        largest_layer = 0
        for module in all_modules:
            layer = module[0]
            if layer[0:5] != 'layer':
                if at_end:
                    end_layers.append(layer)
                else:
                    start_layers.append(layer)

            else:
                at_end = True
                if layer[5] == str(current_hidden):
                    temp.append(layer)
                else:
                    hidden_layers.append(temp)
                    if len(temp) > largest_layer:
                        largest_layer = len(temp)
                    temp = []
                    current_hidden += 1
                    temp.append(layer)
        
        hidden_layers.append(temp)
        if len(temp) > largest_layer:
            largest_layer = len(temp)

        table = np.full((largest_layer, len(hidden_layers)+2), ' '*64)

        # not sure why, but the layers start with an empty string

        table[0:len(start_layers)-1,0] = start_layers[1:]
        columns = ['first layers']

        for i in range(len(hidden_layers)):
            table[0:len(hidden_layers[i]), i+1] = hidden_layers[i]
            columns.append('block ' + str(i+1))

        columns.append('end layers')
        table[0:len(end_layers), -1] = end_layers

        pdf = pd.DataFrame(table, columns=columns)
    
        with pd.option_context('display.max_rows',None):
            display(pdf.style.hide_index())

        

    def _hook_into(self, name):
        """Returns a hook function meant to be registered with register_forward_hook"""
        def hook(model, input, output):
            self._activations[name] = output
        return hook

    #TODO in case we want to apply different weights to different activations, perhaps this should return a dictionary instead
    #TODO not sure if to('cpu') slows us down. is there a way to encapsulate the behavior of _get_activations without this?
        #QUESTION why do we need to clone?
        #QUESTION why do we need to transfer to cpu arent we doing this later anyways?

    #TODO First axis is batch axis. We might consider parametrizing our hooked model so that it works
    # with more than one example in the batch dimension (not relevant right now but might be later)
    # TODO perhaps make it clear that each index in list should be a tuple (channel_idx, w_idx, h_idx) where w_idx and h_idx can be lists themselves.
    # The problem with this approach is that spatial indices will always constitute a regular grid
    # and will always be the same across all feature map activations
    # TODO we could consider just having one integer index array (channel_idxs, w_idxs, h_idxs) instead of using a list to store channel_idxs
    # That way it might also be possible to retrieve different non-grid spatial activations across feature map activations.
    # but be careful when doing numpy integer array indexing, it can be a little iffy.
    # see integer array indexing in https://numpy.org/doc/stable/reference/arrays.indexing.html
    # TODO if we use the integer index array method then we can get a weighted output by using a corresponding weight matrix.
    def _get_activations(self, target_dict):
        """Clones the values returned by the forward hooks and returns them as a list"""
        res = []
        for (name, index) in target_dict.items():
            activation = self._activations[name].clone().to('cpu')
            if isinstance(index, list):
                for i in index:
                    res.append(activation[0][i])
            elif index is not None:
                res.append(activation[0][index]) #as far as i understand the first axis in a tensor contains nothing interesting, so always index past this
            else:
                res.append(activation)
        self._activations.clear()
        return res

    def _register_hooks(self, module_names: t.List[str]):
        """Registers forward hooks on the internal model modules with names in module_names"""
        for module_name in module_names:
            module = self.model.get_submodule(module_name)
            self._hooks[module_name] = module.register_forward_hook(self._hook_into(module_name))

    def _unregister_hooks(self):
        """Removes any registered hooks and clears the interal hook dictionary"""
        for module_name in self._hooks.keys():
            self._hooks[module_name].remove()
        self._hooks.clear()
            
    def forward(self, x: torch.Tensor, target_dict):
        """Runs forward on the internal model and returns activations for any model targets specified.
            target_dict should be a dictionary of valid module names and indices in the internal model.
            Module names can be found by calling HookedModel.show_modules()"""
        self._register_hooks(target_dict.keys())
        x = self.model.forward(x)
        self._unregister_hooks()
        return x, self._get_activations(target_dict)

# TODO these should probably be removed too
class Dreamnet50(Exposed_model):
    def __init__(self, model):
        super().__init__(model, 'fc')
class Googledream(Exposed_model):

    def __init__(self, model):
        super().__init__(model, 'dropout')


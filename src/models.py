import torch
import copy
import torch.nn as nn
from torchvision import models
import typing as t

def get_model(name, pretrained = True, num_classes = None, device = 'cuda'):
    if name == 'resnet18':
        model = models.resnet18(pretrained = pretrained)
    elif name == 'resnet34':
        model = models.resnet34(pretrained = pretrained)
    elif name == 'googlenet':
        model = models.googlenet(pretrained=pretrained)
    else:
        model = models.resnet50(pretrained=pretrained)
    
    if num_classes is not None:
        num_fc = model.fc.in_features
        model.fc = nn.Linear(num_fc, num_classes)
    
    model = model.to(device)

    model.aux_dict = {'name': name, 'pretrained': pretrained, 
                      'num_classes': num_classes, 'train_iters': 0, 'test_acc': None}

    return model

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
        print(list(self.model.named_modules()))

    def _hook_into(self, name):
        """Returns a hook function meant to be registered with register_forward_hook"""
        def hook(model, input, output):
            self._activations[name] = output
        return hook

    #TODO in case we want to apply different weights to different activations, perhaps this should return a dictionary instead
    def _get_activations(self):
        """Clones the values returned by the forward hooks and returns them as a list"""
        res = []
        for val in self._activations.values():
            res.append(val.clone())
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
            
    def forward(self, x: torch.Tensor, targets):
        """"Runs forward on the internal model and returns activations for any model targets specified.
            targets should be a list of valid module names in the internal model.
            Module names can be found by calling HookedModel.show_modules()"""
        self._register_hooks(targets)
        x = self.model.forward(x)
        self._unregister_hooks()
        return x, self._get_activations()

class Dreamnet50(Exposed_model):
    def __init__(self, model):
        super().__init__(model, 'fc')
class Googledream(Exposed_model):

    def __init__(self, model):
        super().__init__(model, 'dropout')
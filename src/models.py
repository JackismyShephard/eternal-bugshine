import torch
import copy
import torch.nn as nn
from torchvision import models

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
    def __init__(self, model):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.aux_dict = model.aux_dict
        self.hooks = {}
        self.hook_results = {}
        self.activations = {}
    
    def show_modules(self):
        print(list(self.model.named_modules()))

    def _get_activation(self, name):
        # the hook signature
        def hook(model, input, output):
            self.activations[name] = output
        return hook

    def register_hook(self, module_string):
        module = self.model.get_submodule(module_string)
        self.hooks[module_string] = module.register_forward_hook(self._get_activation(module_string))

    def register_hooks(self, module_strings):
        for module_string in module_strings:
            module = self.model.get_submodule(module_string)
            self.hooks[module_string] = module.register_forward_hook(self._get_activation(module_string))

    def unregister_hook(self, name):
        self.hooks[name].remove()
        del self.hooks[name]

    def unregister_hooks(self):
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()
            
    def forward(self, x):
        return self.model.forward(x)

class Dreamnet50(Exposed_model):
    def __init__(self, model):
        super().__init__(model, 'fc')
class Googledream(Exposed_model):

    def __init__(self, model):
        super().__init__(model, 'dropout')
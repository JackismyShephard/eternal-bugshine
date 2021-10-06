import torch
import copy
import torch.nn as nn

class Dreamnet50(torch.nn.Module):
    def __str__(self):
        return 'dreamnet50'
    def __init__(self, model):
        super().__init__()
        self.model = copy.deepcopy(model)
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
            if name == 'fc':
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

class Googledream(Dreamnet50):
    def __str__(self):
        return 'googledream'

    def __init__(self, model):
        super().__init__(model)
    
    def forward(self, x, out_info):
        out_activations = {}
        for (name, layer) in self.layers.items():
            if name == 'dropout':
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

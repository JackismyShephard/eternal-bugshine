import torch
import copy
import torch.nn as nn

class Dream_model(torch.nn.Module):
    def __init__(self, model, flatten_layer):
        super().__init__()
        self.model = copy.deepcopy(model)
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

class Dreamnet50(Dream_model):
    def __init__(self, model):
        super().__init__(model, 'fc')
class Googledream(Dream_model):

    def __init__(self, model):
        super().__init__(model, 'dropout')

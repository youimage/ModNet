import torch
import torch.nn as nn

class ModLayer(nn.Module):
    def __init__(self, input_dim, output_dim, mod):
        super(ModLayer, self).__init__()
        self.mod = mod
        self.register_buffer("W", torch.randint(0, mod, (input_dim, output_dim), dtype=torch.int64))

    def forward(self, x):
        return (x @ self.W) % self.mod

class DeepModNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, mod):
        super(DeepModNet, self).__init__()
        self.mod = mod
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(ModLayer(dims[i], dims[i+1], mod))

    def forward(self, x):
        h = x
        for layer in self.layers[:-1]:
            h = layer(h)
        out = self.layers[-1](h)
        return out

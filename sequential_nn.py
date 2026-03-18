import torch
import torch.nn as nn


class SentimentNN(nn.Module):

    def __init__(self, embedding_dimensions, layers):
        super().__init__()

        self.layers = nn.ModuleList()
        previous = embedding_dimensions

        for layer in layers:
            self.layers.append(nn.Linear(previous, layer))
            previous = layer

        self.layers.append(nn.Linear(previous, 1))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))

        x = self.layers[-1](x)
        return x
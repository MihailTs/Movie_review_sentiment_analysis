import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class RecurrentNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths, h=None):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed, h)
        return torch.sigmoid(self.fc(h_n.squeeze(0)))

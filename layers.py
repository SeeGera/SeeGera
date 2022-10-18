import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init_range = np.sqrt(6.0 / (self.input_dim + self.output_dim))
        torch.nn.init.uniform_(self.weight, a=-init_range, b=init_range)

    def forward(self, adj, x):
        x = torch.mm(x, self.weight)
        x = torch.spmm(adj, x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super().__init__()

        self.n_layers = n_layers
        self.dropout_layer = nn.Dropout(p=dropout)
        self.convs = nn.ModuleList()

        if n_layers == 1:
            self.convs.append(GraphConvolution(in_dim, out_dim))
        else:
            self.convs.append(GraphConvolution(in_dim, hid_dim))
            for i in range(n_layers - 2):
                self.convs.append(GraphConvolution(hid_dim, hid_dim))
            self.convs.append(GraphConvolution(hid_dim, out_dim))

    def forward(self, g, inputs):
        outputs = None
        if len(inputs.shape) == 2:  # GCN
            h = inputs
            for l in range(self.n_layers - 1):
                h = self.dropout_layer(h)
                h = F.relu(self.convs[l](g, h))
            h = self.dropout_layer(h)
            h = self.convs[-1](g, h)
            if self.n_layers == 1:
                h = F.relu(h)
            outputs = h
        else:
            assert len(inputs.shape) == 3
            K = inputs.shape[1]
            for i in range(K):
                h = inputs[:, i, :].squeeze(1)
                for l in range(self.n_layers - 1):
                    h = self.dropout_layer(h)
                    h = F.relu(self.convs[l](g, h))
                h = self.dropout_layer(h)
                h = self.convs[-1](g, h)
                if self.n_layers == 1:
                    h = F.relu(h)
                if i == 0:
                    outputs = h.unsqueeze(1)
                else:
                    outputs = torch.cat((outputs, h.unsqueeze(1)), dim=1)
        return outputs


class DenseModel(nn.Module):
    """Stack of fully connected layers."""
    def __init__(self, in_dim, num_hidden, out_dim, num_layers, dropout):
        super(DenseModel, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.dropout_layer = nn.Dropout(p=dropout)

        if num_layers == 1:
            self.layers.append(nn.Linear(in_dim, out_dim))
        else:
            self.layers.append(nn.Linear(in_dim, num_hidden))
            for l in range(1, num_layers - 1):
                self.layers.append(nn.Linear(num_hidden, num_hidden))
            self.layers.append(nn.Linear(num_hidden, out_dim))

    def forward(self, x):
        for l in range(self.num_layers - 1):
            x = self.dropout_layer(x)
            x = self.layers[l](x)
            x = torch.tanh(x)

        x = self.dropout_layer(x)
        x = self.layers[-1](x)

        return x


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class, dropout):
        super().__init__()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, x):
        x = self.dropout_layer(x)
        logits = self.linear(x)
        return logits

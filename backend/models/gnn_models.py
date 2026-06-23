"""GNN architectures for illicit-transaction node classification.

Both models output a single raw logit per node. Loss is applied externally
via BCEWithLogitsLoss (with pos_weight for class imbalance), so these forward
passes intentionally skip the final sigmoid for numerical stability.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x.squeeze(-1)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x.squeeze(-1)

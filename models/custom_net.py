from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import InstanceNorm, BatchNorm
import torch
'''
ADDED NORMALIZATIONS AND CHANGED LEAKY_RELU TO RELU. ADDED ChebConv LAYER
F1 scores achieved on the large dataset
0.736 on Run 1
0.481 on Run 2 
O.420 on Run 3
0.472 on Run 4

F1 scores achieved on the smaller dataset
0.8416 on Run 1
0.5382 on Run 2
0.3906 on Run 3
0.8641 on Run 4
'''
class net(torch.nn.Module):
    def __init__(self, hidden, features, classes, out_features):
        super(net, self).__init__()
        self.hidden_layers = torch.nn.ModuleList()
        self.instNorm = InstanceNorm(in_channels=out_features, affine=True)
        self.batchNorm = BatchNorm(in_channels=out_features)
        self.conv1 = GraphConv(in_channels=features, out_channels=out_features)
        self.conv2 = ChebConv(in_channels=out_features, out_channels=out_features, K=3)
        for layer in range(hidden):
            self.hidden_layers.append(SAGEConv(in_channels=out_features, out_channels=out_features, normalize=True))
        self.final_conv = GraphConv(in_channels=out_features, out_channels=classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.batchNorm(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.relu(x)
        for layer in self.hidden_layers:
             x = layer(x, edge_index)
             x = self.instNorm(x)
             x = torch.nn.functional.relu(x)
        x = self.final_conv(x, edge_index)
        return x

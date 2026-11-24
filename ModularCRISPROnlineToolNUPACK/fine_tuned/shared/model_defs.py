# model_defs.py
import torch
import torch.nn as nn
import torch.nn.functional as F

########################################
# Define CNN1, CNN2, MLP, GNN Branches
########################################

class CNNBranch1(nn.Module):
    """
    CNN branch for (60,4).
    """
    def __init__(self, filters, kernel_size, dense_units):
        super().__init__()
        self.conv = nn.Conv1d(4, filters, kernel_size, padding=(kernel_size-1)//2)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(filters*30, dense_units)
    def forward(self, x):
        x = x.permute(0,2,1)  # (batch,4,30)
        x = F.relu(self.conv(x))
        x = self.pool(x)      # (batch,filters,30)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc(x))
        return x
    
class CNNBranch2(nn.Module):
    """
    CNN branch for (90,3).
    """
    def __init__(self, filters, kernel_size, dense_units):
        super().__init__()
        self.conv = nn.Conv1d(3, filters, kernel_size, padding=(kernel_size-1)//2)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(filters*45, dense_units)
    def forward(self, x):
        x = x.permute(0,2,1)  # (batch,3,45)
        x = F.relu(self.conv(x))
        x = self.pool(x)      # (batch,filters,45)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc(x))
        return x
    
class MLPBranch120(nn.Module):
    """
    A single hidden layer MLP for 120D => output dimension mlp_dim.
    """
    def __init__(self, input_dim=120, hidden_dim=32):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
    def forward(self, x):
        # x shape: (batch,120)
        x = F.relu(self.fc(x))  # (batch,hidden_dim)
        return x
    
from torch_geometric.nn import NNConv, global_mean_pool, global_max_pool

class GNNBranch(nn.Module):
    def __init__(self, hidden_dim=32, edge_attr_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim

        def make_edge_mlp(edge_attr_dim, in_channels, out_channels):
            return nn.Sequential(
                nn.Linear(edge_attr_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, in_channels * out_channels)
            )

        # gRNA
        self.edge_mlp_c   = make_edge_mlp(edge_attr_dim, 4, hidden_dim)
        self.edge_mlp_c2  = make_edge_mlp(edge_attr_dim, hidden_dim, hidden_dim)
        self.conv1_c = NNConv(4, hidden_dim, self.edge_mlp_c, aggr='mean')
        self.conv2_c = NNConv(hidden_dim, hidden_dim, self.edge_mlp_c2, aggr='mean')

        # ssDNA_target_bh
        self.edge_mlp_bh  = make_edge_mlp(edge_attr_dim, 4, hidden_dim)
        self.edge_mlp_bh2 = make_edge_mlp(edge_attr_dim, hidden_dim, hidden_dim)
        self.conv1_t_bh = NNConv(4, hidden_dim, self.edge_mlp_bh, aggr='mean')
        self.conv2_t_bh = NNConv(hidden_dim, hidden_dim, self.edge_mlp_bh2, aggr='mean')

        # duplex
        self.edge_mlp_d   = make_edge_mlp(edge_attr_dim, 4, hidden_dim)
        self.edge_mlp_d2  = make_edge_mlp(edge_attr_dim, hidden_dim, hidden_dim)
        self.conv1_d = NNConv(4, hidden_dim, self.edge_mlp_d, aggr='mean')
        self.conv2_d = NNConv(hidden_dim, hidden_dim, self.edge_mlp_d2, aggr='mean')

        self.mlp_merge = nn.Sequential(
            nn.Linear(hidden_dim * 3 * 2, hidden_dim),  # 3 graphs, mean+max
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def pool(self, x, batch):
        return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

    def forward(self, gRNA_data, ssDNA_target_bh_data, duplex_data):
        def process(graph_data, conv1, conv2):
            x, edge_index, edge_attr, batch = graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch
            x = F.elu(conv1(x, edge_index, edge_attr))   # << correct order
            x = F.elu(conv2(x, edge_index, edge_attr))
            return self.pool(x, batch)

        x_c   = process(gRNA_data, self.conv1_c,   self.conv2_c)
        x_t_bh = process(ssDNA_target_bh_data, self.conv1_t_bh, self.conv2_t_bh)
        x_d   = process(duplex_data, self.conv1_d, self.conv2_d)

        merged = torch.cat([x_c, x_t_bh, x_d], dim=1)
        return self.mlp_merge(merged)
    
########################################
# Final Fusion Model
########################################

class CNN_GNN_MLP_Fusion(nn.Module):
    """
    End-to-end: 
      - CNNBranch1 => feat_cnn1
      - CNNBranch2 => feat_cnn2
      - GNNBranch  => feat_gnn
      - MLPBranch120 => feat_mlp
    Concat => dropout => final FC => 1
    """
    def __init__(self,
                 filters1, kernel_size1, dense_units1,  # CNN1
                 filters2, kernel_size2, dense_units2,  # CNN2
                 gnn_hidden_dim,
                 mlp_hidden_dim,  # single hidden dimension for MLP
                 final_fc_dim,
                 dropout_rate=0.0):  # new hyperparameter for dropout
        super().__init__()

        self.cnn_branch1 = CNNBranch1(filters1, kernel_size1, dense_units1)
        self.cnn_branch2 = CNNBranch2(filters2, kernel_size2, dense_units2)
        self.gnn_branch = GNNBranch(hidden_dim=gnn_hidden_dim)
        self.mlp_branch = MLPBranch120(input_dim=120, hidden_dim=mlp_hidden_dim)
        
        # total dimension = (dense_units1 + dense_units2 + gnn_hidden_dim + mlp_hidden_dim)
        total_dim = dense_units1 + dense_units2 + gnn_hidden_dim + mlp_hidden_dim
        
        self.dropout_rate = dropout_rate
        self.hidden = nn.Linear(total_dim, final_fc_dim)
        self.out = nn.Linear(final_fc_dim, 1)

    def forward(self, gRNA_data, ssDNA_target_bh_data, duplex_data, x1, x2, x3):
        feat_cnn1 = self.cnn_branch1(x1)                   # (batch, dense_units1)
        feat_cnn2 = self.cnn_branch2(x2)                   # (batch, dense_units2)
        feat_gnn = self.gnn_branch(gRNA_data, ssDNA_target_bh_data, duplex_data)  # (batch, gnn_hidden_dim)
        feat_mlp = self.mlp_branch(x3)                 # (batch, mlp_hidden_dim)
        
        merged = torch.cat([feat_cnn1, feat_cnn2, feat_gnn, feat_mlp], dim=1)
        # Apply dropout on the concatenated features
        merged = F.dropout(merged, p=self.dropout_rate, training=self.training)
        x = F.relu(self.hidden(merged))   # (batch, final_fc_dim)
        out = self.out(x)                 # (batch, 1)
        return out.view(-1)

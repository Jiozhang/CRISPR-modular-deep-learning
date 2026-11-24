# model_defs.py
import torch
import torch.nn as nn
import torch.nn.functional as F

########################################
# Define CNN1, CNN2, MLP, GNN Branches
########################################

class CNNBranch1(nn.Module):
    """
    CNN branch for (231,4).
    """
    def __init__(self, filters, kernel_size, dense_units):
        super().__init__()
        self.conv = nn.Conv1d(4, filters, kernel_size, padding=(kernel_size-1)//2)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(filters*115, dense_units)
    def forward(self, x):
        x = x.permute(0,2,1)  # (batch,4,115)
        x = F.relu(self.conv(x))
        x = self.pool(x)      # (batch,filters,115)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc(x))
        return x
    
class CNNBranch2(nn.Module):
    """
    CNN branch for (705,3).
    """
    def __init__(self, filters, kernel_size, dense_units):
        super().__init__()
        self.conv = nn.Conv1d(3, filters, kernel_size, padding=(kernel_size-1)//2)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(filters*352, dense_units)
    def forward(self, x):
        x = x.permute(0,2,1)  # (batch,3,705)
        x = F.relu(self.conv(x))
        x = self.pool(x)      # (batch,filters,352)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc(x))
        return x
    
class MLPBranch41(nn.Module):
    """
    A single hidden layer MLP for 41D => output dimension mlp_dim.
    """
    def __init__(self, input_dim=41, hidden_dim=32):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
    def forward(self, x):
        # x shape: (batch,41)
        x = F.relu(self.fc(x))  # (batch,hidden_dim)
        return x
    
from torch_geometric.nn import NNConv, global_mean_pool, global_max_pool

class GNNBranch(nn.Module):
    def __init__(self, hidden_dim=32, edge_attr_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim

        def make_edge_mlp(edge_attr_dim, in_channels, out_channels):
            return nn.Sequential(
                nn.Linear(edge_attr_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, in_channels * out_channels)
            )

        # gRNA
        self.edge_mlp_c = make_edge_mlp(edge_attr_dim, 4, hidden_dim)
        self.edge_mlp_c2 = make_edge_mlp(edge_attr_dim, hidden_dim, hidden_dim)
        self.conv1_c = NNConv(4, hidden_dim, self.edge_mlp_c, aggr='mean')
        self.conv2_c = NNConv(hidden_dim, hidden_dim, self.edge_mlp_c2, aggr='mean')

        # target_bh
        self.edge_mlp_bh = make_edge_mlp(edge_attr_dim, 4, hidden_dim)
        self.edge_mlp_bh2 = make_edge_mlp(edge_attr_dim, hidden_dim, hidden_dim)
        self.conv1_t_bh = NNConv(4, hidden_dim, self.edge_mlp_bh, aggr='mean')
        self.conv2_t_bh = NNConv(hidden_dim, hidden_dim, self.edge_mlp_bh2, aggr='mean')

        # duplex
        self.edge_mlp_d = make_edge_mlp(edge_attr_dim, 4, hidden_dim)
        self.edge_mlp_d2 = make_edge_mlp(edge_attr_dim, hidden_dim, hidden_dim)
        self.conv1_d = NNConv(4, hidden_dim, self.edge_mlp_d, aggr='mean')
        self.conv2_d = NNConv(hidden_dim, hidden_dim, self.edge_mlp_d2, aggr='mean')

        # target_ah
        self.edge_mlp_ah = make_edge_mlp(edge_attr_dim, 4, hidden_dim)
        self.edge_mlp_ah2 = make_edge_mlp(edge_attr_dim, hidden_dim, hidden_dim)
        self.conv1_t_ah = NNConv(4, hidden_dim, self.edge_mlp_ah, aggr='mean')
        self.conv2_t_ah = NNConv(hidden_dim, hidden_dim, self.edge_mlp_ah2, aggr='mean')

        # triplex_bc
        self.edge_mlp_bc = make_edge_mlp(edge_attr_dim, 4, hidden_dim)
        self.edge_mlp_bc2 = make_edge_mlp(edge_attr_dim, hidden_dim, hidden_dim)
        self.conv1_t_bc = NNConv(4, hidden_dim, self.edge_mlp_bc, aggr='mean')
        self.conv2_t_bc = NNConv(hidden_dim, hidden_dim, self.edge_mlp_bc2, aggr='mean')

        # triplex_ac
        self.edge_mlp_ac = make_edge_mlp(edge_attr_dim, 4, hidden_dim)
        self.edge_mlp_ac2 = make_edge_mlp(edge_attr_dim, hidden_dim, hidden_dim)
        self.conv1_t_ac = NNConv(4, hidden_dim, self.edge_mlp_ac, aggr='mean')
        self.conv2_t_ac = NNConv(hidden_dim, hidden_dim, self.edge_mlp_ac2, aggr='mean')

        # MLP after pooling all graph embeddings
        self.mlp_merge = nn.Sequential(
            nn.Linear(hidden_dim * 6 * 2, hidden_dim),  # 4 graphs, mean+max
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def pool(self, x, batch):
        return torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch)
        ], dim=1)

    def forward(self, gRNA_data, target_bh_data, duplex_data, target_ah_data, triplex_bc_data, triplex_ac_data):
        def process(graph_data, conv1, conv2):
            x, edge_index, edge_attr, batch = (
                graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch
            )
            x = F.elu(conv1(x, edge_index, edge_attr))
            x = F.elu(conv2(x, edge_index, edge_attr))
            return self.pool(x, batch)

        x_c = process(gRNA_data, self.conv1_c, self.conv2_c)
        x_t_bh = process(target_bh_data, self.conv1_t_bh, self.conv2_t_bh)
        x_d = process(duplex_data, self.conv1_d, self.conv2_d)
        x_t_ah = process(target_ah_data, self.conv1_t_ah, self.conv2_t_ah)
        x_t_bc = process(triplex_bc_data, self.conv1_t_bc, self.conv2_t_bc)
        x_t_ac = process(triplex_ac_data, self.conv1_t_ac, self.conv2_t_ac)

        merged = torch.cat([x_c, x_t_bh, x_d, x_t_ah, x_t_bc, x_t_ac], dim=1)
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
      - MLPBranch41 => feat_mlp
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
        self.mlp_branch = MLPBranch41(input_dim=41, hidden_dim=mlp_hidden_dim)
        
        # total dimension = (dense_units1 + dense_units2 + gnn_hidden_dim + mlp_hidden_dim)
        total_dim = dense_units1 + dense_units2 + gnn_hidden_dim + mlp_hidden_dim
        self.dropout_rate = dropout_rate
        self.hidden = nn.Linear(total_dim, final_fc_dim)
        self.out = nn.Linear(final_fc_dim, 1)

    def forward(self, gRNA_data, target_bh_data, duplex_data, target_ah_data, triplex_bc_data, triplex_ac_data, x1, x2, x3):
        feat_cnn1 = self.cnn_branch1(x1)                   # (batch, dense_units1)
        feat_cnn2 = self.cnn_branch2(x2)                   # (batch, dense_units2)
        feat_gnn = self.gnn_branch(gRNA_data, target_bh_data, duplex_data, target_ah_data, triplex_bc_data, triplex_ac_data)  # (batch, gnn_hidden_dim)
        feat_mlp = self.mlp_branch(x3)                 # (batch, mlp_hidden_dim)
        
        merged = torch.cat([feat_cnn1, feat_cnn2, feat_gnn, feat_mlp], dim=1)
        # Apply dropout on the concatenated features
        merged = F.dropout(merged, p=self.dropout_rate, training=self.training)
        x = F.relu(self.hidden(merged))   # (batch, final_fc_dim)
        out = self.out(x)                 # (batch, 1)
        return out.view(-1)
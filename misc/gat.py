# -*- coding: utf-8 -*-
# @Author: Songlin.Zhai
# @Date:   2022-07-05
# @Contact: slinzhai@gmail.com


import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax as dgl_edge_softmax


""" 
Graph Propagation Modules: GAT
"""
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=1,
        feat_drop=0.1, attn_drop=0.1, leaky_relu_alpha=0.2, residual=False):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, num_heads * out_dim, bias=False) # Parameters to be trained
        self.attn_l = nn.Parameter(torch.Tensor(size=(1, num_heads, out_dim))) # Parameters to be trained
        self.attn_r = nn.Parameter(torch.Tensor(size=(1, num_heads, out_dim))) # Parameters to be trained
        nn.init.xavier_normal_(self.fc.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_l.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_r.data, gain=1.414)
        self.num_heads = num_heads
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_alpha)
        self.softmax = dgl_edge_softmax
        self.residual = residual
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
                nn.init.xavier_normal_(self.res_fc.weight.data, gain=1.414)
            else:
                self.res_fc = None

    def forward(self, g:dgl.graph, feature):
        # prepare
        h = self.feat_drop(feature)  # NxD
        ft = self.fc(h).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
        a1 = (ft * self.attn_l).sum(dim=-1).unsqueeze(-1) # N x H x 1
        a2 = (ft * self.attn_r).sum(dim=-1).unsqueeze(-1) # N x H x 1
        g.ndata['ft'] = ft
        g.ndata['a1'] = a1
        g.ndata['a2'] = a2
        # 1. compute edge attention
        g.apply_edges(self.edge_attention)
        # 2. compute softmax
        self.edge_softmax(g)
        # 3. compute the aggregated node features scaled by the dropped,
        # unnormalized attention values.
        g.update_all(fn.src_mul_edge('ft', 'a_drop', 'ft'), fn.sum('ft', 'ft'))
        ret = g.ndata['ft']
        # 4. residual
        if self.residual:
            if self.res_fc is not None:
                resval = self.res_fc(h).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
            else:
                resval = torch.unsqueeze(h, 1)  # Nx1xD'
            ret = resval + ret
        return ret

    def edge_attention(self, edges):
        # an edge UDF to compute unnormalized attention values from src and dst
        a = self.leaky_relu(edges.src['a1'] + edges.dst['a2'])
        return {'a' : a}

    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a'))
        # Dropout attention scores and save them
        g.edata['a_drop'] = self.attn_drop(attention)

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, heads,
                       feat_drop=0.1, attn_drop=0.1, residual=False,
                       activation=F.leaky_relu,  leaky_relu_alpha=0.2):
        super(GAT, self).__init__()
        self.gat_layers = nn.ModuleList() # Parameters to be trained
        self.num_layers = num_layers
        self.activation = activation
        # Input layer, no residual
        self.gat_layers.append(GATLayer(in_dim, hidden_dim, heads[0], feat_drop, attn_drop, leaky_relu_alpha, False))
        # Hidden layers, due to multi-head, the in_dim = hidden_dim * num_heads
        for l in range(1, num_layers-1):
            self.gat_layers.append(GATLayer(hidden_dim * heads[l-1], hidden_dim, heads[l], feat_drop, attn_drop, leaky_relu_alpha, residual))
        # Output layer
        self.gat_layers.append(GATLayer(hidden_dim * heads[-2], out_dim, heads[-1], feat_drop, attn_drop, leaky_relu_alpha, residual))

    def forward(self, g:dgl.graph, features):
        h = features
        for l in range(self.num_layers-1):
            h = self.gat_layers[l](g, h).flatten(1)
            h = self.activation(h)
        # Output projection
        h = self.gat_layers[-1](g, h).mean(1)
        return h
    
    def map_feats(self, feats):
        h = feats
        for layer in self.gat_layers:
            h = layer.feat_drop(h)
            h = layer.fc(h)
        return h


if __name__ == '__main__':
    g=dgl.graph(([1,1,2,3,4,4,5],[2,3,3,4,5,6,7]))
    feats=torch.tensor([[0,0,0,0],[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4],[5,5,5,5],[6,6,6,6],[7,7,7,7.]])
    feats_2 = torch.rand((8,64))
    gat_ = GAT(64,128,128,2,[4,1],F.relu)
    print(gat_)
    gat__feats = gat_(g.subgraph([0,1,2,3,4,5,6,7]), feats_2)
    print(gat__feats.size())
    print(gat__feats)
    g.ndata['h'] = feats
    print(dgl.mean_nodes(g, 'h'))
    print(g.ndata['h'])
    #print(gat__feats.size())
    #print(gat__feats)
    print('\nNew features...')
    print(gat_.map_feats(feats_2))
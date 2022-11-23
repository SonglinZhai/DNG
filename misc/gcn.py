# -*- coding: utf-8 -*-
# @Author: Songlin.Zhai
# @Date:   2022-07-05
# @Contact: slinzhai@gmail.com


import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import math
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax as dgl_edge_softmax


""" 
Graph Propagation Modules: GCN
"""
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout, bias=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.dropout = nn.Dropout(p=dropout)
        if bias: self.bias = nn.Parameter(torch.Tensor(out_feats))
        else: self.bias = None
        if activation: self.activation = activation
        else: self.activation = None
        self.init_parameters()

    def init_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h):
        h = self.dropout(h)
        h = torch.mm(h, self.weight)
        # normalization by square root of src degree
        #h = h * g.ndata['norm'] # comment by sonnglin
        g.ndata['h'] = h
        g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h'))
        h = g.ndata.pop('h')+h  # '+h' is added by songlin
        # normalization by square root of dst degree
        h = h * g.ndata['norm']
        print(g.ndata['norm'])
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, activation, in_dropout=0.1, hidden_dropout=0.1, output_dropout=0.0):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(in_dim, hidden_dim, activation, in_dropout))
        # hidden layers
        for l in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim, activation, hidden_dropout))
        # output layer
        self.layers.append(GCNLayer(hidden_dim, out_dim, None, output_dropout))

    def forward(self, g, features):
        h = features
        degs = g.in_degrees().float()
        degs = degs + 1 # added by songlin
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(h.device)
        g.ndata['norm'] = norm.unsqueeze(1)
        for layer in self.layers:
            h = layer(g, h)
        return h
    
    def map_feats(self, feats):
        h = feats
        for layer in self.layers:
            h = layer.dropout(h)
            h = torch.mm(h, layer.weight)
        return h


if __name__ == '__main__':
    g=dgl.graph(([1,1,2,3,4,4,5],[2,3,3,4,5,6,7]))
    g.add_nodes(1)
    print(g.num_nodes())
    feats=torch.tensor([[0,0,0,0],[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4],[5,5,5,5],[6,6,6,6],[7,7,7,7.],[8,8,8,8.]])
    #norm = torch.tensor([[0,0,0,0],[1,1,1,1],[2,2,2,2],[3,3,3,3],[2,2,2,2],[2,2,2,2],[2,2,2,2],[2,2,2,2.]])
    
    #gcn=GCNLayer(4,4,F.leaky_relu_,0.1)
    gcn = GCN(4,8,4,2,F.leaky_relu)
    gcn_feats = gcn(g, feats)
    print(gcn_feats.size())
    print(gcn_feats)
    #print(g.ndata['norm'])
    #print(g.edges())
    #print(g.nodes())
    #print(gcn.map_feats(feats))

    #batch_g = dgl.batch([g,g])
    #feats = torch.cat([feats, feats])
    #print(dgl.unbatch(batch_g))
    #gcn = GCN(4,8,8,2,F.relu)
    #gcn_feats = gcn(batch_g, feats)
    #gcn_feats = gcn_feats.view(2, 8, 8)
    #print(gcn_feats[0] == gcn_feats[1])
    #print(gcn_feats)

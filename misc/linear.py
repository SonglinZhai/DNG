# -*- coding: utf-8 -*-
# @Author: Songlin.Zhai
# @Date:   2022-05-20
# @Contact: slinzhai@gmail.com


import torch

class LinearLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, dropout=False, activation=False):
        super(LinearLayer, self).__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_normal_(self.fc.weight.data, gain=1.414)
        if dropout: self.dropout = torch.nn.Dropout(dropout)
        else: self.dropout = dropout
        self.activation = activation
    
    def forward(self, feats):
        if self.dropout: feats = self.dropout(feats)
        feats = self.fc(feats)
        if self.activation: feats = self.activation(feats)
        return feats

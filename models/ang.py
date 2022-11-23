# -*- coding: utf-8 -*-
# @Author: Songlin.Zhai
# @Date:   2022-05-20
# @Contact: slinzhai@gmail.com


import torch
import torch.nn.functional as F
import numpy as np
from joblib import parallel_backend, Parallel, delayed

from base import BaseModel
from misc import Decompose, LinearLayer


class ANGModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self._config_ = config
        self._decompose_ = Decompose()
        self._in_dim_ = config["in_dim"]
        self._hidden_dim_ = config["hidden_dim"]
        self._out_dim_ = config["out_dim"]
        self._mapping_ = torch.nn.ModuleList()
        self._mapping_.append(LinearLayer(self._in_dim_, self._hidden_dim_, dropout=0.1, activation=F.leaky_relu))
        self._mapping_.append(LinearLayer(self._hidden_dim_, self._hidden_dim_, dropout=0.1, activation=F.leaky_relu))
        self._mapping_.append(LinearLayer(self._hidden_dim_, self._out_dim_, dropout=0.1))

    def config(self, key):
        assert key in self._config_.keys(),\
               print(f"{key} is not in trainner config!")
        return self._config_[key]
    
    def cuda(self):
        self._mapping_ = self._mapping_.cuda()
    
    def to(self, device:torch.device):
        self._mapping_ = self._mapping_.to(device)

    def forward(self, feats:np.array, batch_paths:list, backend='loky', n_jobs=4):
        """ Get prediction for each node based on paths

        Parameters
        ----------
        batch_paths : list(list)
            length = batch_size in training
            length = batch_size*seed_num_path in testing
            Paths of the sub-graph in correct node order
        
        feats : numpy.array
            (batch_size, max_num_nodes_in_path, feature_dim) in training
            (batch_size*seed_num_path, max_num_nodes_in_path, feature_dim) in testing
        
        Return
        ----------
        Best anchors in each path for the query node
        """
        '''
        # feature with shape [batchsize, max_node_num, in_dim]
        with parallel_backend(backend=backend, n_jobs=n_jobs):
            res = Parallel()(delayed(self._decompose_.estimate_inherit_factor)\
                #(self._embeds_[path].transpose(1,0),list(range(len(path)))[:-1],-1)\ # no feats provided
                (feats[idx,:len(path)].transpose(1,0),list(range(len(path)))[:-1],-1)\
                #for path in batch_paths) # no feats provided
                for idx, path in enumerate(batch_paths))
        '''

        # feature with shape [-1, in_dim], flatten all features
        num_n = list(map(len, batch_paths))
        idx_ranges = []
        start = 0
        for n in num_n:
            idx_ranges.append((start,start+n))
            start = start+n
        with parallel_backend(backend=backend, n_jobs=n_jobs):
            res = Parallel()(delayed(self._decompose_.estimate_inherit_factor)\
                    (feats[idx[0]:idx[1]].transpose(1,0))\
                    for idx in idx_ranges
                  )
        predictions = list(map(lambda adj, path: path[np.argmax(adj)], res, batch_paths))
        return predictions

    def get_inherit_factors(self, feats:np.array, batch_paths:list, backend='loky', n_jobs=4):
        """ Get inheritance factors for each node based on paths

        Parameters
        ----------
        batch_paths : list(list)
            length = batch_size in training
            Paths of the sub-graph in correct node order
        
        feats : numpy.array
            (batch_size, max_num_nodes_in_path, feature_dim) in training        
        Return
        ----------
        Inheritance factors
        """
        '''
        # feature with shape [batchsize, max_node_num, in_dim]
        with parallel_backend(backend=backend, n_jobs=n_jobs):
            res = Parallel()(delayed(self._decompose_.estimate_adj_mx_deflation)\
                (feats[idx,:len(path)].transpose(1,0))\
                for idx, path in enumerate(batch_paths))
        seed_adj_factors = list(map(lambda adj: adj[:-1], res))
        query_adj_factors = list(map(lambda adj: adj[-1], res))
        '''

        # feature with shape [-1, in_dim], flatten all features
        num_n = list(map(len, batch_paths))
        idx_ranges = []
        start = 0
        for n in num_n:
            idx_ranges.append((start,start+n))
            start = start+n
        with parallel_backend(backend=backend, n_jobs=n_jobs):
            res = Parallel()(delayed(self._decompose_.estimate_adj_mx_deflation)\
                (feats[idx[0]:idx[1]].transpose(1,0))\
                for idx in idx_ranges)
        seed_adj_factors = list(map(lambda adj: adj[:-1,:-1].flatten(), res))
        query_adj_factors = list(map(lambda adj: adj[-1], res))
        return seed_adj_factors, query_adj_factors

    def map_raw_features(self, raw_feats):
        feats = raw_feats
        for layer in self._mapping_:
            feats = layer(feats)
        return feats


    
    def get_raw_features(self, batch_paths):
        """ Retrieve node features from nn.Embedding

        Parameters
        ----------
        batch_paths : list(list)
            (batch_size, ...)
            Paths of the sub-graph in correct node order
        
        Return
        ----------
        Raw features of nodes
        """
        max_len = max(list(map(len, batch_paths)))
        raw_feats = np.zeros((len(batch_paths), max_len, self._in_dim_))
        for idx, path in enumerate(batch_paths):
            raw_feats[idx,:len(path)] = self._embeds_[path]
        return raw_feats
    
    def _compute_mix_features(self, adj_mx, source, mask):
        """ Compute one node mix feature by mixing matrix and sources

        Parameters
        ----------
        adj_mxs : np.array
            (num_nodes_in_path(different), num_nodes_in_path(different))
            Adjacent matrixs between nodes in each path
        
        source : np.array
            (num_nodes_in_path, feature_dim)
            Independent features computed by Decompose class
        
        mask : np.matrix
            (nodes_in_path(different), nodes_in_path(different))
            Matrixs to mask the incorent value in each adjacent matrix
        
        Return
        ----------
        Node feature computed by the mixing matrix and sources
        (num_nodes_in_path, feature_dim)
        """
        # Pad the incorrect adj_value with zero in the adjacent matrixs
        adj_mx = np.ma.array(adj_mx, mask=mask, fill_value=0.).filled()
        # Compute inverse of the padded adjacent matrixs (i.e., mixing matrix)
        adj_mx = np.linalg.pinv(np.identity(adj_mx.shape[0])-adj_mx)
        #
        # Torch version
        #adj_mx = adj_mx.masked_fill_(torch.from_numpy(mask),0.)
        #adj_mx = torch.pinverse(torch.eye(adj_mx.size(0))-adj_mx)
        #
        # Return the node features computed by the mixing matrixs and sources
        #return torch.matmul(adj_mx, source)
        return np.matmul(adj_mx, source)
    
    def get_mix_features(self, batch_paths, batch_masks, backend='loky', n_jobs=4):
        """ Predict node features by mixing matrix and sources

        Parameters
        ----------
        batch_paths : list(list)
            (batch_size, ...)
            Paths of the sub-graph in correct node order
        
        batch_masks : list(np.matrix)
            (batch_size, num_nodes_in_path(different), num_nodes_in_path(different))
            Matrixs to mask the incorent value in each adjacent matrix

        Return
        ----------
        Mixing features of nodes
        """
        max_len = max(list(map(len, batch_paths)))
        mix_features = np.zeros((len(batch_paths), max_len, self._in_dim_))
        for idx, path in enumerate(batch_paths):
            path_len = len(path)
            # Get source and adjacent matrix
            source, adj_mx = self._decompose_.estimate_source(
                self._embeds_[path].transpose(1,0), max_iter=150*path_len
            )
            # Get mix feature
            mix_features[idx,:path_len] = self._compute_mix_features(
                adj_mx, source.T, batch_masks[idx]
            )
        return mix_features

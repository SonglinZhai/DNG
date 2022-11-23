# -*- coding: utf-8 -*-
# @Author: Songlin.Zhai
# @Date:   2022-05-20
# @Contact: slinzhai@gmail.com

import dgl
import networkx as nx
from random import sample
from numpy import dtype
import torch
import pickle
import os
from tqdm import tqdm
from torch.utils.data import Dataset

from dataset import Taxonomy

class DataSet(Dataset):
    def __init__(self):
        self._info_ = None
        self._g_full_nx_ = None
        self._g_seed_leaf_dgl_ = None
        self._seed_paths_ = None
        self._nodes_ = list()
        self._positions_ = list()
        self._paths_with_nodes_ = list()
        self._paths_without_nodes_ = list()
    
    def __str__(self):
        return self._info_

    def __len__(self):
        return len(self._nodes_)

    def __getitem__(self, idx):
        """ Generate an data instance.
        
        One data instance is a list of 
            (batch_idx, node_id, anchors, graph_path, mask_adjacent_matrix).
        """
        res = list()
        path_idx = sample(list(range(len(self._paths_with_nodes_[idx]))), k=1)[0]
        g_path = self._paths_with_nodes_[idx][path_idx]
        g_path_ = self._paths_without_nodes_[idx][path_idx]
        node_id = self._nodes_[idx]
        res.append(idx)
        res.append(node_id)
        res.append(self._positions_[idx])
        res.append(self._g_seed_leaf_dgl_.subgraph(g_path_)) # for seed_graph
        res.append(self._g_seed_leaf_dgl_.subgraph(g_path))  # for seed_graph+node
        res.append(g_path_)
        res.append(g_path)
        res.append(nx.adjacency_matrix(self._g_full_nx_, g_path_).todense().T) # for seed_ajd
        res.append(nx.adjacency_matrix(self._g_full_nx_, g_path).todense().T)  # for full_ajd
        res.append(nx.adjacency_matrix(self._g_full_nx_, g_path).todense().T[-1])  # for query_dis
        # idx, node_id, anchors,
        # dgl_without_node, dgl_with node,
        # path_without_node, path_with_node,
        # adj_without_node, adj_with_node,
        # adj_query
        return tuple(res)

    def _train_path_(self, graph:nx.DiGraph, node, seed_paths:list):
        """ Generate train path instance,
            append train node into the path that
            contains the node anchors
        
        Parameters
        ----------
        graph : nx.DiGraph()
            full graph of taxonommy
        
        node : node of nx.DiGraph()
            train node id
        
        seed_paths : list(list(node_ids))
            paths parsed from seed taxonomy
        
        Return
        ----------
        paths and anchors of current node
        """
        paths = list()
        #paths_dgl = list()
        paths_ = list()
        #paths_dgl_ = list()
        anchors = [edge[0] for edge in graph.in_edges(node)]
        for seed_path in seed_paths:
            intersection = set(anchors)&set(seed_path)
            if len(intersection) != 0:
                #path_dgl = dgl_graph.subgraph(seed_path+[node])
                #edges = path_dgl.in_edges(path_dgl.num_nodes()-1)
                #eids = path_dgl.edge_ids(edges[0], edges[1])
                #path_dgl.remove_edges(eids)
                paths.append(seed_path+[node])
                #paths_dgl.append(dgl_graph.subgraph(seed_path+[node]))
                paths_.append(seed_path)
                #paths_dgl_.append(dgl_graph.subgraph(seed_path))
        return paths_, paths, anchors
    
    def _test_path_(self, graph:nx.DiGraph, node, seed_paths:list):
        """ Generate test path instance,
            append each node into all paths
        
        Parameters
        ----------
        graph : nx.DiGraph()
            full graph of taxonommy
        
        node : node of nx.DiGraph()
            test node id
        
        seed_paths : list(list(node_ids))
            paths parsed from seed taxonomy
        
        Return
        ----------
        All paths and anchors of current node
        """
        paths = list()
        #paths_dgl = list()
        paths_ = list()
        #paths_dgl_ = list()
        anchors = [edge[0] for edge in graph.in_edges(node)]
        for seed_path in seed_paths:
            #paths_dgl_.append(dgl_graph.subgraph(seed_path))
            #path_dgl = dgl_graph.subgraph(seed_path+[node])
            #edges = path_dgl.in_edges(path_dgl.num_nodes()-1)
            #eids = path_dgl.edge_ids(edges[0], edges[1])
            #path_dgl.remove_edges(eids)
            #paths_dgl.append(path_dgl)
            paths_.append(seed_path)
            #paths_dgl_.append(dgl_graph.subgraph(seed_path))
            paths.append(seed_path+[node])
            #paths_dgl.append(dgl_graph.subgraph(seed_path+[node]))
        return paths_, paths, anchors

    def hold(self, taxo:Taxonomy, mode:str):
        """ Let this dataset hold a taxonomy under train/valid/test mode
        
        Parameters
        ----------
        taxo : a instance of class Taxonomy
             The data this dataset holds
            
        mode : str
              train/valid/test
        """
        print(f"\n---> Building {taxo._name_} {mode} dataset <---")
        # Distill data
        print(f"Distill {mode} nodes ...")
        if mode == 'train':
            self._nodes_ = taxo._train_node_ids_
        elif mode == 'valid':
            self._nodes_ = taxo._valid_node_ids_
        elif mode == 'test':
            self._nodes_ = taxo._test_node_ids_
        else: raise Exception('{} is not the right mode, \
                    change to train, valid or test'.format(mode))
        
        # Assign data and information
        self._info_ = f"Holding {mode} data of {taxo._name_} taxonomy."
        self._taxonomy_ = taxo
        self._seed_paths_ = taxo._seed_paths_
        self._g_full_nx_ = taxo._g_full_
        self._g_seed_leaf_dgl_ = dgl.from_networkx(taxo._g_full_)
        self._g_seed_leaf_dgl_.ndata['init_feats'] = torch.from_numpy(taxo._embeds_,dtype=torch.float)
        print('Removing edges of nodes ...')
        for node in self._nodes_:
            edges = self._g_seed_leaf_dgl_.in_edges(node)
            eids = self._g_seed_leaf_dgl_.edge_ids(edges[0], edges[1])
            self._g_seed_leaf_dgl_.remove_edges(eids)
        # Generate paths with appending the node
        print(f"Generate candidates {mode} paths ...")
        if mode == 'train': process_function = self._train_path_
        else: process_function = self._test_path_
        for node in tqdm(self._nodes_, ncols=60):
            path_, path, anchor = process_function(self._g_full_nx_, node, self._seed_paths_)
            self._positions_.append(anchor)
            self._paths_with_nodes_.append(path)
            self._paths_without_nodes_.append(path_)
        print("Checking data...")
        self.check()
    
    def check(self):
        assert self._g_seed_leaf_dgl_.num_nodes() == self._g_full_nx_.number_of_nodes(),\
            f"Error: dgl_graph.num_nodes()={self._g_seed_leaf_dgl_.num_nodes()},\
              nx_graph.num_nodes()={self._g_full_nx_.number_of_nodes()}"
        assert self._g_seed_leaf_dgl_.num_edges() != self._g_full_nx_.number_of_edges(),\
            f"Error: dgl_graph.num_edges()={self._g_seed_leaf_dgl_.num_edges()},\
              nx_graph.num_edges()={self._g_full_nx_.number_of_edges()}"
        assert len(self._paths_with_nodes_) == len(self._paths_without_nodes_),\
            "Error: len(with_nodes) != len(without_nodes)"
        for p_w, p_wo in zip(self._paths_with_nodes_,self._paths_without_nodes_):
            if len(p_w)==len(p_wo): continue
            else: print("Error")
    
    def save(self, save_fp):
        with open(save_fp, 'wb') as outfstream:
            pickle.dump(self.__dict__, outfstream, 2)

    def load(self, load_fp):
        if not os.path.exists(load_fp):
            print('Taxonomy does not exist in %s'%load_fp)
            exit(0)
        with open(load_fp, 'rb') as infstream:
            # using saved_data = pickle.load(infstream), saved_data['field']
            self.__dict__.update(pickle.load(infstream))

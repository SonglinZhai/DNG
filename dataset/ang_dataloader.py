# -*- coding: utf-8 -*-
# @Author: Songlin.Zhai
# @Date:   2022-05-20
# @Contact: slinzhai@gmail.com

import dgl
import numpy as np

from base import BaseDataLoader
from dataset import DataSet


class ANGDataLoader(BaseDataLoader):
    def __init__(self, config:dict, dataset:DataSet, mode:str):
        assert mode in ['train', 'valid', 'test'], f'Error mode: {mode}'
        if mode == 'train': collate_fn = self._train_fn_
        else: collate_fn = self._test_fn_
        super(ANGDataLoader, self).__init__(
            dataset=dataset, batch_size=config['batch_size'],
            shuffle=config['shuffle'], collate_fn=collate_fn
        )
        self._config_ = config
    
    # idx, node_id, anchors,
    # dgl_without_node, dgl_with node,
    # path_without_node, path_with_node,
    # adj_without_node, adj_with_node,
    # adj_query
    def _train_fn_(self, samples):
        #
        batch_path = list(map(lambda data: data[6], samples))
        batch_seed_adj = list(map(lambda data: data[7].flatten().tolist()[0], samples))
        batch_qurery_adj = list(map(lambda data: data[9].flatten().tolist()[0], samples))
        #batch_mask = list(map(lambda data: ~np.matrix(data, dtype=bool), batch_adj_mx))
        #
        batch_graph = dgl.batch(list(map(lambda data: data[4], samples)))
        return tuple([batch_path, batch_graph, batch_seed_adj, batch_qurery_adj])
    
    # idx, node_id, anchors,
    # dgl_without_node, dgl_with node,
    # path_without_node, path_with_node,
    # adj_without_node, adj_with_node,
    # adj_query
    def _test_fn_(self, samples):
        #max_len_path = max(list(map(lambda data: len(data[3]), samples)))
        #batch_idxs = list(map(lambda data: data[0], samples))
        batch_node_ids = list(map(lambda data: data[1], samples))
        batch_anchor_ids = list(map(lambda data: data[2], samples))
        batch_path = list(map(lambda data: self.dataset._paths_with_nodes_[data[0]], samples))
        #batch_adj_mx = list(map(lambda data: data[4], samples))
        return tuple([batch_node_ids, batch_anchor_ids, batch_path])
    
    def config(self, key):
        assert key in self._config_,\
            f"{key} is not in dataloader config!"
        return self._config_[key]
    
    def size(self):
        return self.dataset.__len__()

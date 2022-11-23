# -*- coding: utf-8 -*-
# @Author: Songlin.Zhai
# @Date:   2022-05-20
# @Contact: slinzhai@gmail.com

import os
import random
import pickle
import itertools
import networkx as nx
import numpy as np
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize

from base import BaseTaxonomy, Node


class Taxonomy(BaseTaxonomy):
    def __init__(self):
        """
        Dataset class for MAG-data, WordNet-data, etc
        """
        self._name_ = None
        self._root_ = None
        self._nodes_ = list()
        self._embeds_ = None
        self._g_full_ = nx.DiGraph()
        self._g_seed_ = None
        self._seed_paths_ = list()
        self._paths_merge_idx_ = list()
        self._paths_merge_flag_ = list()
        self._bfs_node_ids_ = list()
        self._node_id2tx_id_ = dict()
        self._tx_id2node_id_ = dict()
        self._train_node_ids_ = list()
        self._valid_node_ids_ = list()
        self._test_node_ids_ = list()
    
    def __str__(self):
        return f"Taxonomy of '{self._name_}'."

    def _exist_partion_(self, dir_path):
        train_node_file_name = os.path.join(dir_path, f"{self._name_}.terms.train")
        validation_node_file_name = os.path.join(dir_path, f"{self._name_}.terms.valid")
        test_file_name = os.path.join(dir_path, f"{self._name_}.terms.test")

        print("Loading existing train/validation/test partitions")
        raw_train_node_list = self._load_node_list_(train_node_file_name)
        raw_valid_node_list = self._load_node_list_(validation_node_file_name)
        raw_test_node_list = self._load_node_list_(test_file_name)

        self._train_node_ids_ = [self._tx_id2node_id_[tx_id] for tx_id in raw_train_node_list]
        self._validation_node_ids_ = [self._tx_id2node_id_[tx_id] for tx_id in raw_valid_node_list]
        self._test_node_ids_ = [self._tx_id2node_id_[tx_id] for tx_id in raw_test_node_list]

    def _random_partion_(self, nx_graph):
        print("Randomly create train/valid/test nodes ...")
        leaf_node_ids = self._get_leaves_(nx_graph)
        random.seed(47)
        random.shuffle(leaf_node_ids)
        validation_size = int(len(leaf_node_ids) * 0.1)
        test_size = int(len(leaf_node_ids) * 0.1)
        self._valid_node_ids_ = leaf_node_ids[:validation_size]
        self._test_node_ids_ = leaf_node_ids[validation_size:(validation_size+test_size)]
        self._train_node_ids_ = [node_id for node_id in leaf_node_ids\
            if node_id not in self._valid_node_ids_ and node_id not in self._test_node_ids_]

    def _load_node_list_(self, node_list_fp):
        """
        Import node list from external file

        Parameters
        ----------
        node_list_fp : str
            The node list file path
            with format: a 'taxo_id' per line
        """
        node_list = []
        with open(node_list_fp, "r", encoding="utf-8") as ifstream:
            for line in ifstream:
                line = line.strip()
                if line: node_list.append(line)
        return node_list

    def _seed_graph_(self, nx_graph:nx.DiGraph):
        """ Find the seed graph for a given graph
            the seed graph is the graph after
            removing all leaves of the given graph
        
        Parameters
        ----------
        nx_graph : networkx.DiGraph()
            the graph
        
        Return
        ----------
        The seed graph
        """
        nx_graph_copy = nx_graph.copy()
        nx_graph_copy.remove_nodes_from(self._get_leaves_(nx_graph))
        nx_graph_copy.remove_nodes_from(list(nx.isolates(nx_graph)))
        #for node in self._get_leaves_(nx_graph):
            #nx_graph_copy.remove_node(node)
            # Generate the seed graph in dgl package
            #edges = self.g_seed.in_edges(node)
            #eids = self.g_seed.edge_ids(edges[0], edges[1])
            #self.g_seed.remove_edges(eids)
        return nx_graph_copy

    def _seed_path_(self, nx_graph, source, target):
        """ Find all paths between a source
            and a target node in a given graph
        
        Parameters
        ----------
        nx_graph : networkx.DiGraph()
            the graph
        
        source : node of networkx.DiGraph()
            source node
        
        target : node of networkx.DiGraph()
            target node
        
        Return
        ----------
        A list of paths with line/sub-graph mode
        """
        def __graph_order__(nx_graph, path_list):
            # Construct a sub-graph
            #g = nx.DiGraph()
            #for path in path_list: nx.add_path(g, path)
            nodes = list(itertools.chain(*path_list))
            # Return graph topology order as the graph path
            return list(nx.topological_sort(nx_graph.subgraph(nodes)))
        line_paths = list()
        for path in nx.all_simple_paths(nx_graph, source, target):
            line_paths.append(path)
        graph_path = __graph_order__(nx_graph, line_paths)
        return line_paths, graph_path

    def _seed_path_merge_(self, nx_graph):
        """ Generate the partion index of a graph
        
        Parameters
        ----------
        nx_graph : networkx.DiGraph()
            the graph needing partion
        
        Return
        ----------
        Original Leaves for generation graph_paths
        And the partion index for merging prediction when test
        
        Example
        ----------
        For a graph holing edges:
        (0, 1),  (0, 2),  (0, 3),  (1, 4),  (1, 5),
        (2, 6),  (2, 7),  (3, 8),  (3, 9),  (4, 10),
        (5, 10), (5, 11), (5, 12), (5, 13), (6, 14),
        (7, 14), (8, 15), (8, 16), (9, 16), (9, 17)
        
        The dfs_postorder_nodes will be:
        [10, 4, 11, 12, 13, 5, 1, 14, 6, 7, 2, 15, 16, 8, 17, 9, 3, 0]

        Original leaves: -> Loop-1
        [10, 11, 12, 13, 14, 15, 16, 17]
        divided by:
        [4, 5, 6, 7, 8, 9]
        with partion index:
        [0, 1, 4, 5, 5, 7, 8]
        generating group:
        [(10)_[0:1], (11, 12, 13)_[1:4], (14)_[4:5], ()_[5:5], (15, 16)_[5:7], (17)_[7:8]]

        After that, leaves will be: -> Loop-2
        [4, 5, 6, 7, 8, 9]
        divided by:
        [1, 2, 3]
        with partion index:
        [0, 2, 4, 6]
        generating group:
        [(4, 5)_[0:2], (6, 7)_[2:4], (8, 9)_[4:6]]

        Then, leaves will be:  -> Loop-3
        [1, 2, 3]
        divided by:
        [0]
        with partion index:
        [0, 3]
        generating group:
        [(1, 2, 3)_[0:3]]

        Return will be:
        [10, 11, 12, 13, 14, 15, 16, 17]
        [
            [0, 1, 4, 5, 5, 7, 8],
            [0, 2, 4, 6],
            [0, 3]
        ]

        """
        # Find root
        root = self._get_root_(nx_graph)
        assert len(root) == 1, f"Number of root in taxonomy is {len(root)}!"
        root = root[0]
        # All nodes order by dfs traversing in post-order;
        # as such, the descendants of the same ancestor could be placed continuously
        # and the descendants of different ancestors could be divided by an ancestor;
        # then, we can employ 'divide-and-rule' to vote for the best node,
        # the vote procedure can be carried out in the 'bottom-to-up' manner
        res_leaves = list()
        res_partion = list()
        res_partion_flag = list()
        nx_graph_copy = nx_graph.copy()
        # Obtain all nodes order by dfs
        node_order_dfs_full = list(nx.dfs_postorder_nodes(nx_graph, source=root))
        # Get original leaves
        leaves = [node for node in node_order_dfs_full\
                       if self._is_leaf_(nx_graph_copy, node)]
        res_leaves = leaves
        while nx_graph_copy.number_of_nodes() > 1:
            # Remove leaves
            nx_graph_copy.remove_nodes_from(leaves)
            # Treat leaves of new graph as the partion flags
            partition_flag = [node for node in node_order_dfs_full\
                                   if self._is_leaf_(nx_graph_copy, node)]
            # If there is no partion, it means there is only a root in the graph
            if len(partition_flag) == 0: partition_flag.append(root)
            one_level_partion = [0]
            leaf_num = 0
            # Parse pation index
            for node in node_order_dfs_full:
                # Have a partion
                if node in partition_flag:
                    one_level_partion.append(leaf_num)
                # Get a leaf
                if node in leaves: leaf_num = leaf_num + 1
            res_partion.append(one_level_partion)
            res_partion_flag.append(partition_flag)
            # Treat partion as new leaves
            leaves = partition_flag
        return res_leaves, res_partion, res_partion_flag

    def _add_root_(self, nx_graph, np_embeds, root_given=None):
        res = list()
        # Process the root node
        root = self._get_root_(nx_graph)
        if len(root) != 1:
            if root_given != None:
                root_str = root_given
                root_node_id = nx_graph.number_of_nodes()
                root_taxo_id = root_given+'_given_'+str(root_node_id)
                root_embed = np_embeds[root].mean(axis=0)
                edges = [(root_node_id, cd) for cd in root]
                res = [root_str, root_node_id, root_taxo_id, root_embed, edges]
            else: raise Exception("Provide a root node!")
        # root_str, root_node_id, root_taxo_id,
        # root_embed, root_edges
        return res

    def _cycles_edges_(self, nx_graph):
        cycles = list(nx.simple_cycles(nx_graph))
        cycle_edges = [(path[-1],path[0]) for path in cycles]
        return cycle_edges

    def import_taxo(self, config):
        """
        Import taxonomy from external file

        Parameters
        ----------
        config: dict
            The config dict of a taxonomy data

        Parameters will be retrieve from config:
        dir_path : str
            The path to a directory:
            (1).containing three input files: *.term, *.taxo and *.terms.embed
            or/and
            (2).containing three files: *.term.train, *.term.valid and *.term.test
        
        existing_partition : bool optional
            whether to use the existing the train/validation/test partitions
            or randomly sample new ones, by default False
        """
        self._name_ = config['name'] 
        dir_path = config['data_dir']
        dset = os.path.join(dir_path, f"{self._name_}.dset")
        if os.path.exists(dset):
            print(f'Loading data from {dset} ...')
            self.load(dset)
            return self
        # Load from files
        print("\n---> Import Taxonomy <---")
        node_fname = os.path.join(dir_path, f"{self._name_}.terms")
        edge_fname = os.path.join(dir_path, f"{self._name_}.taxo")
        embed_fname = os.path.join(dir_path, f"{self._name_}.terms.embed")
        # Load nodes, edges and embeddings from file
        print("Loading nodes and edges ...")
        tx_id2node = self._import_nodes_(node_fname)
        self._tx_id2node_id_ = {tx_id:idx for idx, tx_id in enumerate(tx_id2node.keys()) } 
        self._node_id2tx_id_ = {v:k for k, v in self._tx_id2node_id_.items()}
        # Generate vocab
        # tx_id is the old taxo_id read from {self.name}.terms file,
        # node_id is the new taxo_id from 0 to len(vocab)
        self._nodes_ = [tx_id2node[self._node_id2tx_id_[node_id]]
                      for node_id in range(len(tx_id2node))]
        print("Creat graph ...")
        self._g_full_.add_nodes_from(list(self._node_id2tx_id_.keys()))
        edges = self._import_edges_(edge_fname)
        for edge in edges:
            self._g_full_.add_edge(self._tx_id2node_id_[edge[0]], self._tx_id2node_id_[edge[1]])
        self._g_full_.remove_edges_from(self._cycles_edges_(self._g_full_))
        print("Loading embeddings ...")
        embeds = KeyedVectors.load_word2vec_format(embed_fname)
        _embeds_ = np.zeros(embeds.vectors.shape)
        for node_id, tx_id in self._node_id2tx_id_.items():
            _embeds_[node_id] = embeds[tx_id]
        
        print("Check graph ...")
        #print(nx.is_directed_acyclic_graph(self._g_full_))
        #print(self._g_full_.has_node(self._g_full_.number_of_nodes()))
        #print(nx.number_of_isolates(self._g_full_))
        #print(list(nx.isolates(self._g_full_)))
        if len(self._get_root_(self._g_full_)) != 1:
            # root_str, root_node_id, root_taxo_id,
            # root_embed, root_edges
            print(f"Adding a root node ({config['added_root']}) ...")
            res = self._add_root_(self._g_full_, _embeds_, config['added_root'])
            # node_id and tx_id must not be in two dicts
            assert res[2] not in self._tx_id2node_id_.keys(),\
                f"New tx_id {res[2]} have been in self._tx_id2node_id_"
            assert res[1] not in self._node_id2tx_id_.keys(),\
                f"New tx_id {res[1]} have been in self._node_id2tx_id_"
            self._root_ = res[1]
            self._nodes_.append(Node(res[2], res[0]))
            self._tx_id2node_id_[res[2]] = res[1]
            self._node_id2tx_id_[res[1]] = res[2]
            self._embeds_ = np.zeros((_embeds_.shape[0]+1, _embeds_.shape[1]))
            self._embeds_[:_embeds_.shape[0]] = _embeds_
            self._embeds_[res[1]] = res[3]
            self._g_full_.add_edges_from(res[4])
        else:
            self._root_ = self._get_root_(self._g_full_)[0]
            self._embeds_ = _embeds_
        
        if config['normalize_embed']:
            self._embeds_ = normalize(self._embeds_, norm='l2', axis=1)

        # Generate train/valid/test nodes
        if config['existing_partition']: self._exist_partion_(dir_path)
        else: self._random_partion_(self._g_full_)
        # Parse the seed graph (without train/valid/test nodes)
        print("Parse the seed taxonomy ...")
        self._g_seed_ = self._seed_graph_(self._g_full_)
        print("Parse paths of the seed taxonomy ...")
        leaves, partions, partion_flag = self._seed_path_merge_(self._g_seed_)
        self._paths_merge_idx_ = partions
        self._paths_merge_flag_ = partion_flag
        for leaf in leaves:
            # Only store graph path
            self._seed_paths_.append(self._seed_path_(self._g_seed_, self._root_, leaf)[1])
        self._bfs_node_ids_ = list(nx.topological_sort(self._g_seed_))#list(nx.bfs_tree(self._g_seed_, root))

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

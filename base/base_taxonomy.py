# -*- coding: utf-8 -*-
# @Author: Songlin.Zhai
# @Date:   2022-05-20
# @Contact: slinzhai@gmail.com

from abc import abstractmethod


class Node(object):
    def __init__(self, tx_id, name="none"):
        """ Class for one node

        Parameters
        ----------
        tx_id : str
            raw taxonomy id, eg: 134652429
        
        name : str
            name of taxonomy node, eg: computer science
        """
        self._tx_id_ = tx_id
        self._name_ = name
        
    def __str__(self):
        return "Node {} Name: {}".format(self._tx_id_, self._name_)


class BaseTaxonomy(object):
    """
    Base class for all taxonomy classes
    """
    def __init__(self):
        pass

    @abstractmethod
    def import_taxo(self, *input):
        raise NotImplementedError
    
    def _import_nodes_(self, node_fp):
        """
        Import nodes of taxonomy from external file

        Parameters
        ----------
        node_fp : str
            The node file path of input file
            with format: taxon1_id \t taxon1_surface_name
        """
        tx_id2node = {}
        # load nodes
        with open(node_fp, "r", encoding="utf-8") as ifstream:
            for line in ifstream:
                line = line.strip()
                if line:
                    segs = line.split("\t")
                    assert len(segs) == 2, f"Wrong number of segmentations {line}"
                    node = Node(tx_id=segs[0], name=segs[1])
                    tx_id2node[segs[0]] = node
        return tx_id2node
    
    def _import_edges_(self, edge_fp):
        """
        Import edges of taxonomy from external file

        Parameters
        ----------
        edge_fp : str
            The edge file path of input file
            with format: taxon1_id \t taxon2_id
        """
        edges = list()
        with open(edge_fp, "r", encoding="utf-8") as ifstream:
            for line in ifstream:
                line = line.strip()
                if line:
                    segs = line.split("\t")
                    assert len(segs) == 2, f"Wrong number of segmentations {line}"
                    parent_taxon = segs[0]
                    child_taxon = segs[1]
                    edges.append([parent_taxon, child_taxon])
        return edges
    
    def _get_root_(self, nx_graph):
        """ Find all roots of a given graph
        
        Parameters
        ----------
        nx_graph : networkx.DiGraph()
            the graph
        
        Return
        ----------
        A list of all root nodes
        """
        root = list()
        for node in nx_graph.nodes():
            if nx_graph.in_degree(node) == 0 and\
               nx_graph.out_degree(node) != 0: root.append(node)
        return root
    
    def _get_leaves_(self, nx_graph):
        """ Find all leaves of a given graph
        
        Parameters
        ----------
        nx_graph : networkx.DiGraph()
            the graph
        
        Return
        ----------
        A list of leaf nodes
        """
        leaves = list()
        for node in nx_graph.nodes():
            if nx_graph.in_degree(node) != 0 and\
               nx_graph.out_degree(node) == 0: leaves.append(node)
        return leaves
    
    def _is_leaf_(self, nx_graph, node):
        """ Whether a node is a leaf in a given graph
        
        Parameters
        ----------
        nx_graph : networkx.DiGraph()
            the graph
        
        node : node of networkx.DiGraph()

        Return
        ----------
        A bool value
        """
        res = False
        if node in nx_graph.nodes():
            if nx_graph.out_degree(node)==0 and\
               nx_graph.in_degree(node)!=0: res=True
        return res

    def _is_root_(self, nx_graph, node):
        """ Whether a node is root in a given graph
        
        Parameters
        ----------
        nx_graph : networkx.DiGraph()
            the graph
        
        node : node of networkx.DiGraph()

        Return
        ----------
        A bool value
        """
        res = False
        if node in nx_graph.nodes():
            if nx_graph.out_degree(node)!=0 and\
               nx_graph.in_degree(node)==0: res=True
        return res

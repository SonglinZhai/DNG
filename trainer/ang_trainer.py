# -*- coding: utf-8 -*-
# @Author: Songlin.Zhai
# @Date:   2022-05-20
# @Contact: slinzhai@gmail.com

import torch
import itertools
import numpy as np
import torch.nn.functional as F
import json

from base import BaseTrainer
from misc import find_nodes_order, read_json, write_json
#from logger import WriterTensorboardX


class ANGTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, config, train_data_loader, eval_data_loader,
                       model, loss, optimizer, metrics, logger):
        super(ANGTrainer, self).__init__(config, model, logger)
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.loss = loss['loss']
        self.loss_type = loss['name']
        self.optimizer = optimizer['optim']
        self.optim_type = optimizer['name']
        self.metrics = metrics
        # Setup visualization writer instance
        #self.writer = WriterTensorboardX(self.train_cfg["log_dir"], self.logger, self.train_cfg['tensorboardX'])
        #self.writer.add_text('Text', 'Model Architecture: {}'.format(self.config['arch']), 0)
        #self.writer.add_text('Text', 'Training Data Loader: {}'.format(self.config['train_data_loader']), 0)
        #self.writer.add_text('Text', 'Loss Function: {}'.format(self.config['loss']), 0)
        #self.writer.add_text('Text', 'Optimizer: {}'.format(self.config['optimizer']), 0)

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        log_step = int(np.sqrt(self.train_data_loader.config('batch_size')))
        for batch_idx, batch_data in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()
            raw_feats = batch_data[1].ndata['init_feats']
            if self.device != 'cpu': raw_feats = raw_feats.cuda()
            raw_feats = self.model.map_raw_features(raw_feats) # only this step using gpu
            seed_adj_pred, query_adj_pred = self.model.get_inherit_factors(raw_feats.detach().cpu().numpy(), batch_data[0])
            # flatten the unequal ajd_mx to a 1-D array
            # the loss does not be changed
            # however, if padding these adj_mx into equal length,
            # the loss will be changed !
            seed_adj_pred = torch.tensor(list(itertools.chain(*seed_adj_pred)))
            query_adj_pred = torch.tensor(list(itertools.chain(*query_adj_pred)))
            seed_adj_target = torch.tensor(list(itertools.chain(*batch_data[2]))).type_as(seed_adj_pred)
            query_adj_target = torch.tensor(list(itertools.chain(*batch_data[3]))).type_as(query_adj_pred)
            if self.device != 'cpu':
                seed_adj_pred = seed_adj_pred.cuda()
                query_adj_pred = query_adj_pred.cuda()
                seed_adj_target = seed_adj_target.cuda()
                query_adj_target = query_adj_target.cuda()
            loss = 0.5*self.loss(seed_adj_target, seed_adj_pred) +\
                   0.5*self.loss(query_adj_target, query_adj_pred)
            loss.backward()
            self.optimizer.step()
            #self.writer.set_step((epoch - 1) * len(self.train_data_loader) + batch_idx)
            #self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            # Save log information
            if batch_idx % log_step == 0:
                self.logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.train_data_loader.config('batch_size'),
                    self.train_data_loader.size(),
                    100.0 * batch_idx / self.train_data_loader.size(),
                    loss.item())
                )
        log = {'loss': total_loss / self.train_data_loader.size()}
        return log

    def _test_epoch(self, taxonomy, epoch):
        """
        Evaluation after training an epoch

        :return: A log that contains information about evaluation

        Note:
            The evaluation metrics in log must have the key 'val_metrics'.
        """
        # Data used in evaluation
        merge_idxs = taxonomy._paths_merge_idx_
        merge_flags = taxonomy._paths_merge_flag_
        taxo_root = taxonomy._root_
        bfs_nodes = taxonomy._bfs_node_ids_
        init_embeds= taxonomy._embeds_
        # Start evaluation
        self.model.eval()
        predictions = dict()
        anchors = dict()
        with torch.no_grad():
            for batch_data in self.eval_data_loader:
                raw_feats = torch.from_numpy(init_embeds[list(itertools.chain(*batch_data[2][0]))])
                if self.device != 'cpu': raw_feats = raw_feats.cuda()
                raw_feats = self.model.map_raw_features(raw_feats)
                max_node_ids_bf = self.model(raw_feats, batch_data[2][0])
                #predictions[batch_data[0]] = [max_node_ids_bf]
                predictions[batch_data[0][0]] = list()
                anchors[batch_data[0][0]] = batch_data[1][0]
                # merging initial predictions for the best one
                for merge_idx, merge_flag in zip(merge_idxs, merge_flags):
                    once_merge = list()
                    for idx in range(1, len(merge_idx)):
                        merge_start = merge_idx[idx-1]
                        merge_end = merge_idx[idx]
                        if merge_end > merge_start:
                            merge_nodes = max_node_ids_bf[merge_start:merge_end]
                            # Add root node and the split node
                            merge_nodes = find_nodes_order(set(merge_nodes+[merge_flag[idx-1],taxo_root]), bfs_nodes)
                            # Add current node
                            once_merge.append(merge_nodes+[batch_data[0][0]])
                    raw_feats = torch.from_numpy(init_embeds[list(itertools.chain(*once_merge))])
                    if self.device != 'cpu': raw_feats = raw_feats.cuda()
                    raw_feats = self.model.map_raw_features(raw_feats)
                    max_node_ids_af = self.model(raw_feats, once_merge)
                    # append the current merging result into prediction dict
                    #predictions[batch_data[0]].append(max_node_ids_af)
                    predictions[batch_data[0][0]].append(list(set(max_node_ids_bf)-set(max_node_ids_af)))
                    max_node_ids_bf = max_node_ids_af
                predictions[batch_data[0][0]].append(list(set(max_node_ids_af)))
        metrics = self.metrics(predictions, anchors)
        return metrics
    
    def save(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        """
        state = {
            'model_type': self.model_type,
            'model': self.model.state_dict(),
            'optim_type': self.optim_type,
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }
        filename = str(self.config('save_dir') / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def load(self, ckpt_path):
        """
        Resume from saved checkpoints

        :param ckpt_path: Checkpoint path to be resumed
        """
        resume_path = str(ckpt_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)

        # load architecture params from checkpoint.
        if checkpoint['model_type'] != self.model_type:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['model'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['optim_type'] != self.optim_type:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else: self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(checkpoint['epoch']+1))



    def _test_all(self, taxonomy):
        # Data used in evaluation
        merge_idxs = taxonomy._paths_merge_idx_
        merge_flags = taxonomy._paths_merge_flag_
        taxo_root = taxonomy._root_
        taxo_name = taxonomy._name_
        bfs_nodes = taxonomy._bfs_node_ids_

        self.logger.debug("Generating data...")
        g_full = self.eval_data_loader.dataset._taxonomy_._g_full_
        all_leaves = self.eval_data_loader.dataset._taxonomy_._get_leaves_(g_full)
        all_paths = self.eval_data_loader.dataset._taxonomy_._seed_paths_
        all_g_paths = list()
        for node in all_leaves:
            one_g_path = list()
            for one_path in all_paths:
                one_g_path.append(one_path+[node])
            all_g_paths.append(one_g_path)
        all_anchors = list()
        for node in all_leaves:
            all_anchors.append([edge[0] for edge in g_full.in_edges(node)])
        
        predictions = dict()
        anchors = dict()
        for node, anchor, path in zip(all_leaves, all_anchors, all_g_paths):
            self.logger.debug("Prediction for node {}".format(node))
            raw_feats = torch.from_numpy(self.model.get_raw_features(path))
            if self.device != 'cpu': raw_feats = raw_feats.cuda()
            raw_feats = self.model.map_raw_features(raw_feats)
            max_node_ids_bf = self.model(raw_feats, path)
            predictions[node] = list()
            anchors[node] = anchor
            # merging initial predictions for the best one
            for merge_idx, merge_flag in zip(merge_idxs, merge_flags):
                once_merge = list()
                for idx in range(1, len(merge_idx)):
                    merge_start = merge_idx[idx-1]
                    merge_end = merge_idx[idx]
                    if merge_end > merge_start:
                        merge_nodes = max_node_ids_bf[merge_start:merge_end]
                        # Add root node and the split node
                        merge_nodes = find_nodes_order(set(merge_nodes+[merge_flag[idx-1],taxo_root]), bfs_nodes)
                        # Add current node
                        once_merge.append(merge_nodes+[node])
                raw_feats = torch.from_numpy(self.model.get_raw_features(once_merge))
                if self.device != 'cpu': raw_feats = raw_feats.cuda()
                raw_feats = self.model.map_raw_features(raw_feats)
                max_node_ids_af = self.model(raw_feats, once_merge)
                # append the current merging result into prediction dict
                #predictions[batch_data[0]].append(max_node_ids_af)
                predictions[node].append(list(set(max_node_ids_bf)-set(max_node_ids_af)))
                max_node_ids_bf = max_node_ids_af
            predictions[node].append(list(set(max_node_ids_af)))
        write_json(predictions, f'all_predictions_{taxo_name}.json')
        write_json(anchors, f'all_anchors_{taxo_name}.json')

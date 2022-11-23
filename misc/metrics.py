# -*- coding: utf-8 -*-
# @Author: Songlin.Zhai
# @Date:   2022-05-20
# @Contact: slinzhai@gmail.com

import math
import numpy as np


class Metric(object):
    def __init__(self, metrics):
        """
        metrics:
        hit@1, hit@3 ...
        precision@1, precision@3 ...
        recall@1, recall@3 ...
        mrr,
        mr
        """
        self._all_metrics_ = ['hit', 'recall', 'precision', 'mr', 'mrr']
        self._metrics_ = dict()
        self._parse_metrics_(metrics)
    
    def __call__(self, predictions, anchors):
        # ranks: worst rank, ranks_: best rank
        ranks,ranks_ = self._get_ranks_(predictions, anchors)
        res = dict()
        if 'hit' in self._metrics_:
            for k in self._metrics_['hit']:
                res['hit@'+str(k)+'_worst'] = self._hit_at_k_(ranks, k)
                res['hit@'+str(k)+'_best'] = self._hit_at_k_(ranks_, k)
        if 'precision' in self._metrics_:
            for k in self._metrics_['precision']:
                res['pre@'+str(k)+'_worst'] = self._precision_at_k_(ranks, k)
                res['pre@'+str(k)+'_best'] = self._precision_at_k_(ranks_, k)
        if 'recall' in self._metrics_:
            for k in self._metrics_['recall']:
                res['rec@'+str(k)+'_worst'] = self._recall_at_k_(ranks, k)
                res['rec@'+str(k)+'_best'] = self._recall_at_k_(ranks_, k)
        if 'mr' in self._metrics_:
            res['mr_worst'] = self._mr_(ranks)
            res['mr_best'] = self._mr_(ranks_)
        if 'mrr' in self._metrics_:
            res['mrr_worst'] = self._mrr_(ranks)
            res['mrr_best'] = self._mrr_(ranks_)
        return res
    
    def _parse_metrics_(self, metrics):
        for one_metric_ in metrics:
            one_metric = one_metric_.split('@')
            one_metric[0] = one_metric[0].lower()
            assert one_metric[0] in self._all_metrics_,\
                f"{one_metric[0]} is not implemented in current metrics!"
            if one_metric[0] not in self._metrics_:
                self._metrics_[one_metric[0]] = list()
            if len(one_metric) > 1:
                assert one_metric[1].isdigit(),\
                    f"{one_metric_} is error format!"
                self._metrics_[one_metric[0]].append(int(one_metric[1]))
    
    def _get_ranks_(self, predictions:dict, anchors:dict):
        ranks = list()
        ranks_ = list()
        for node_id, anchor in anchors.items():
            rank = list()
            rank_ = list()
            predict = predictions[node_id]
            for one_anchor in anchor:
                sum_rank = 0
                sum_rank_ = 1
                position = 0
                position_ = 0
                for idx in range(1, len(predict)+1):
                    sum_rank = sum_rank + len(predict[-idx])
                    if one_anchor in predict[-idx]:
                        position = sum_rank
                        position_ = sum_rank_
                        break
                    sum_rank_ = sum_rank_ + len(predict[-idx])
                if position != 0:
                    rank.append(position)
                    rank_.append(position_)
                else:
                    rank.append(sum_rank)
                    rank_.append(sum_rank_)
            assert len(anchor)==len(rank), "Error when generating ranks!"
            assert len(anchor)==len(rank_), "Error when generating ranks!"
            ranks.append(rank)
            ranks_.append(rank_)
        return ranks, ranks_

    def _precision_at_k_(self, all_ranks, k):
        tp_num = np.zeros((len(all_ranks)), dtype=np.float32)
        for idx, one_rank in enumerate(all_ranks):
            tp_num[idx] = len(list(filter(lambda x: x <= k, one_rank)))
        res = tp_num/k
        return res.mean()
    
    def _recall_at_k_(self, all_ranks, k):
        tp_num = np.zeros((len(all_ranks)), dtype=np.float32)
        true_num = np.zeros((len(all_ranks)), dtype=np.float32)
        for idx, one_rank in enumerate(all_ranks):
            tp_num[idx] = len(list(filter(lambda x: x <= k, one_rank)))
            true_num[idx] = len(one_rank)
        res= tp_num/true_num
        return res.mean()
    
    def _mr_(self, all_ranks):
        rank_positions = np.zeros((len(all_ranks)), dtype=np.float32)
        for idx, one_rank in enumerate(all_ranks):
            rank_positions[idx] = sum(one_rank)/len(one_rank)
        return rank_positions.mean()
    
    def _mrr_(self, all_ranks):
        """
        Scaled MRR score,
        Check eq. (2) in the PinSAGE paper: [https://arxiv.org/pdf/1806.01973.pdf]
        """
        scaled_rank_positions = np.zeros((len(all_ranks)), dtype=np.float32)
        for idx, one_rank in enumerate(all_ranks):
            scaled_rank_positions[idx] = sum([1.0/(math.ceil(x/10)) for x in one_rank])/len(one_rank)
        return scaled_rank_positions.mean()


if __name__ == '__main__':
    metric = Metric(['Precision@1s0', 'precision@5', 'recall@3'])
    print(metric._metrics_)

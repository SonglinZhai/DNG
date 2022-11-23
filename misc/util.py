# -*- coding: utf-8 -*-
# @Author: Songlin.Zhai
# @Date:   2022-05-20
# @Contact: slinzhai@gmail.com

import json
import torch
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
import numpy as np
import difflib
from sklearn.metrics.pairwise import cosine_similarity

class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()


def exponential_dist(value:torch.tensor, lamda:float=1.0):
    exp_dist_func = torch.distributions.exponential.Exponential(lamda)
    # CDF = 1 - e^{-lambda*x}    for x >= 0;  CDF = 0 for x <  0
    # PDF = lambda*e^{-lambda*x} for x >  0;  PDF = 0 for x <= 0
    prob = exp_dist_func.cdf(value)*(-1)+(1.0)
    return prob*lamda

def have_nan(np_arr):
    return True in np.isnan(np_arr)

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def load_config(config_fp):
    with open(config_fp, 'r', encoding='utf-8') as ifstream:
        config = json.load(ifstream, object_hook=OrderedDict)
    return config

def read_json(json_fp):
    global null
    null = ' '
    dicts = dict()
    with open(json_fp, 'r', encoding='utf-8') as ifstream:
        dicts = eval(ifstream.read())
    return dicts

def write_json(dict_lst, json_fp):
    with open(json_fp, 'w', encoding='utf-8') as json_file:
        json.dump(dict_lst, json_file, indent = 4, ensure_ascii=False)

def find_nodes_order(nodes, raw_list):
    res = list()
    for node in raw_list:
        if node in nodes:
            res.append(node)
    assert len(res) == len(nodes),\
        f"Error when looking up node orders..."
    return res

def string_sim(str1, str2):
    return difflib.SequenceMatcher(None, str1, str2).quick_ratio()

def similarity(filename):
    vectors = list()
    names = list()
    with open(filename, 'r', encoding='UTF-8') as fin:
        for line in fin:
            fileds = line.strip('\n').split()
            vectors.append(fileds[1:])
            names.append(fileds[0])
    vectors = np.asarray(vectors, dtype=np.float)
    print(vectors.shape)
    '''
    sims = list()
    for i in range(len(vectors)):
        cur_sim = list()
        for j in range(len(vectors)):
            cur_sim.append(format(cosine_similarity(np.reshape(vectors[i], newshape=[1,-1]),np.reshape(vectors[j], newshape=[1,-1]))[0][0], '.4f'))
        sims.append(cur_sim)
    print(sims)
    '''
    sims = cosine_similarity(vectors)

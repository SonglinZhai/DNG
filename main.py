import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore','Using backend: pytorch')
import networkx as nx
import torch

from dataset import Taxonomy, DataSet, ANGDataLoader
from models import ANGModel
from logger import Logger
from trainer import ANGTrainer
from misc import Metric, Loss, load_config


if __name__ == "__main__":
    config = load_config('config.json')
    taxo = Taxonomy()
    #taxo.import_taxo(config['mag_cs'])
    taxo.load('computer_science.dset')
    print(len(taxo._nodes_))
    print(len(taxo._node_id2tx_id_))
    print(len(taxo._node_id2tx_id_))
    print(nx.number_of_isolates(taxo._g_full_))
    print(nx.number_of_isolates(taxo._g_seed_))
    print(taxo._g_full_.number_of_nodes())
    print(taxo._g_seed_.number_of_nodes())
    print(len(taxo._seed_paths_))
    print(len(taxo._paths_merge_idx_))
    print(len(taxo._paths_merge_flag_))

    dset_train = DataSet()
    print('Reach here!')
    dset_train.hold(taxo, 'train')
    dset_train.save('computer_science.train.dset')
    dset_load_train = ANGDataLoader(config['data_loader']['train_args'], dset_train, mode='train')
    dset_test = DataSet()
    dset_test.hold(taxo, 'test')
    dset_test.save('computer_science.test.dset')
    dset_load_test = ANGDataLoader(config['data_loader']['test_args'], dset_test, mode='test')
    model = ANGModel(config['arch']['ang_args'])
    optimizer = torch.optim.Adam(model.parameters(),
        lr=config['optimizer']['args']['lr'],
        weight_decay=config['optimizer']['args']['weight_decay']
    )
    metrics = Metric(config['metrics'])
    loss = Loss(config['loss'])
    logger = Logger(config['log'], 'test')
    logger.set_verbosity(config['trainer']['verbosity'])
    trainer = ANGTrainer(config['trainer'],
        dset_load_train, dset_load_test,
        {'name': 'ang', 'model':model},
        {'name': 'kl_diversity', 'loss':loss},
        {'name': 'adam', 'optim':optimizer},
        metrics, logger
    )
    log_train = trainer._train_epoch(1)
    log_test = trainer._test_epoch(1)
    print(log_train)
    print(log_test)

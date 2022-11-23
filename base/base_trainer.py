# -*- coding: utf-8 -*-
# @Author: Songlin.Zhai
# @Date:   2022-05-20
# @Contact: slinzhai@gmail.com

import torch
from abc import abstractmethod
from numpy import inf


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, config, model, logger):
        self._config_ = config
        self.logger = logger
        self.model = model['model']
        self.model_type = model['name']
        # Setup GPU device if available, move model into configured device
        device_ids = self._prepare_device_(config['gpu_ids'])
        if len(device_ids) == 0:
            self.device = 'cpu'
            self.model.to(torch.device('cpu'))
        elif len(device_ids) == 1:
            self.device = 'cuda:'+str(device_ids[0])
            self.model.to(torch.device('cuda:'+str(device_ids[0])))
        else:
            self.device = 'cuda:'+','.join(str(x) for x in device_ids)
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

    def _prepare_device_(self, gpu_ids_use):
        """
        setup GPU device if available, move model into configured device
        """
        devices = list()
        n_gpu = torch.cuda.device_count()
        if n_gpu == 0:
            if len(gpu_ids_use) > 0:
                self.logger.warning("Warning: There\'s no GPU available on this "
                                    "machine, training will be performed on CPU.")
        else:
            for gpu_id in gpu_ids_use:
                if gpu_id > n_gpu:
                    self.logger.warning("Warning: GPU {} does not exist, and but there are "
                           " only {} GPUs available on this machine.".format(gpu_id, n_gpu))
                else: devices.append(gpu_id)
        return devices

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError
    
    @abstractmethod
    def _test_epoch(self, epoch):
        """
        Evaluation logic after an training epoch

        :param epoch: Current training epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        for epoch in range(1, self.config('epochs')+1):
            result = self._train_epoch(epoch)
            # Save log information into log dict
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value
            # Print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))
    
    def config(self, key):
        assert key in self._config_.keys(),\
               print(f"{key} is not in trainner config!")
        return self._config_[key]

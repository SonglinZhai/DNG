# -*- coding: utf-8 -*-
# @Author: Songlin.Zhai
# @Date:   2022-05-20
# @Contact: slinzhai@gmail.com


from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size,
                 shuffle=True, collate_fn=default_collate):
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'collate_fn': collate_fn,
            'shuffle': shuffle
        }
        super(BaseDataLoader, self).__init__(**self.init_kwargs)
    
    def __str__(self):
        return self.init_kwargs

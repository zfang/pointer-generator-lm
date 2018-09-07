import torch


class DataParallel(torch.nn.DataParallel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item == 'module':
            return self._modules['module']
        return self.module.__getattribute__(item)

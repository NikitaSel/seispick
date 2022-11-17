import os
import numpy as np
from functools import partial
from ...dataset import PickingDataSet


class TimePickingDataSet(PickingDataSet):
    def __init__(self, path: str, transforms: list[callable] = None):
        super().__init__(path, transforms)

    def _load_inp_tar(self, inp_path: str, tar_path: str):
        inp_tar = {'input': None, 'target': None}

        load_txt = partial(np.loadtxt, dtype='float64', delimiter=',')

        inp_tar['input'] = load_txt(inp_path)
        inp_tar['target'] = load_txt(tar_path)

        inp_tar['input'] = inp_tar['input'][np.newaxis, :]
        inp_tar['target'] = np.array([inp_tar['target']], dtype=int)[np.newaxis, :]

        return inp_tar

class ResultDataset(TimePickingDataSet):
     def __getitem__(self, key):
        inp_path, tar_path = self._inp_paths[key], self._tar_path[key]

        inp_tar = self._load_inp_tar(inp_path, tar_path)

        if self.transforms is not None:
            transformed = self.transforms(image=inp_tar['input'], mask=inp_tar['target'])
            inp_tar['input'], inp_tar['target'] = transformed['image'], transformed['mask']

        inp_tar_info = inp_tar
        inp_tar_info['info'] = os.path.split(self._inp_paths[key])[1]
        return inp_tar_info

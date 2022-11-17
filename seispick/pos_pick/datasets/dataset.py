import os
import json
import numpy as np
from typing import Optional
from functools import partial
from torch.utils.data import Dataset
from ...utils.helper import Target
from ...dataset import PickingDataSet


class PositionPickingDataSet(PickingDataSet):
    def __init__(self, 
                 path: str, 
                 transforms: Optional[list[callable]] = None):
        super(PositionPickingDataSet, self).__init__(path, transforms)

    def _load_inp_tar(self, inp_path: str, tar_path: str) -> dict:
        inp_tar = {'input': None, 'target': None}

        load_txt = partial(np.loadtxt, dtype='float64', delimiter=',')
        inp_tar['input'] = load_txt(inp_path)

        with open(tar_path, "r") as f:
            inp_tar['target'] = json.load(f)

        for key in inp_tar['target']:
            inp_tar['target'][key] = np.array(inp_tar['target'][key])

        tar = Target(**inp_tar['target'])
        inp_tar['target'] = tar

        return inp_tar

import os
import json
import numpy as np
from ..readsgy import Data
from ..utils.helper import Target
from ..savedata import BaseSaveData


class SaveSeiLines(BaseSaveData):
    def __init__(self,
                 path_to: str,
                 data: Data, 
                 min_len: int,
                 transforms: list[callable] = None, 
                 p: int = 0.15):
        super(SaveSeiLines, self).__init__(path_to, data, transforms, p)
        self.min_len = min_len

    def save(self):
        self._make_paths()

        for station in self.data.stations:
            for line in self.data[station].lines:
                st_line = self.data[station][line].stack_points()

                inp = st_line['traces']
                tar = Target(st_line['sx'], st_line['sy'],
                             st_line['gx'], st_line['gy'],
                             st_line['rge'], st_line['sd'],
                             st_line['tr_interval'], st_line['drt'])

                if self.transforms is not None:
                    inp, tar = self.transforms(inp, tar)

                path_to = self.get_train_or_test_path()

                if (inp is not None) and (tar is not None):
                    if inp.shape[0] >= self.min_len:
                        inputs_path = os.path.join(path_to, self.inputs_folder_name)
                        targets_path = os.path.join(path_to, self.targets_folder_name)

                        np.savetxt(os.path.join(inputs_path, f'{station}_{line}'), inp, delimiter=',')

                        with open(os.path.join(targets_path, f'{station}_{line}.json'), "w") as f:
                            json.dump(self._target_obj_to_dict(tar), f)

    def _target_obj_to_dict(self, tar: Target) -> dict[list]:
        tar2dict = {
                    'sx': tar.sx.tolist(), 'sy': tar.sy.tolist(),
                    'gx': tar.gx.tolist(), 'gy': tar.gy.tolist(),
                    'rge': tar.rge.tolist(), 'sd': tar.sd.tolist(),
                    'tr_int': tar.tr_int.tolist(), 'drt': tar.drt.tolist()
              }
        return tar2dict

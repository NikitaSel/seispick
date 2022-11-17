import os
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
from ..readsgy import Data, Point
from ..plotsgy import plot
from ..utils.helper import Target 
from ..utils.utils import make_path
from ..savedata import BaseSaveData


class SaveSeiSignals(BaseSaveData):
    def __init__(self,
                 path_to: str,
                 data: Data,
                 V: int,
                 transforms: Optional[list[callable]] = None, 
                 p: int = 0.15,
                 save_img: bool=True):

        super(SaveSeiSignals, self).__init__(path_to, data, transforms, p)

        self.V = V
        self.save_img = save_img
        if save_img:
            self.image_folder_name = 'images'

    def save(self) -> None:
        self._make_paths()
        if self.save_img:
            train_path = os.path.join(self.path, 'train')
            test_path = os.path.join(self.path, 'test')

            _ = make_path(train_path, self.image_folder_name)
            _ = make_path(test_path, self.image_folder_name)

        for point in  self.data:
            data = point.data

            inp = data['traces']
            tar = Target(data['sx'], data['sy'],
                         data['gx'], data['gy'],
                         data['rge'], data['sd'],
                         data['tr_interval'], data['drt'])

            if self.transforms is not None:
                inp, tar = self.transforms(inp, tar)

            if (inp is not None) and (tar is not None):

                # if point.info['Point'] % 2 == 0 or point.info['Station'] != 1807:
                #     continue
                
                if point.info['Point'] % 2 == 0:
                    continue

                path_to = self.get_train_or_test_path()

                inp_path = os.path.join(path_to, self.inputs_folder_name)
                tar_path = os.path.join(path_to, self.targets_folder_name)

                np.savetxt(os.path.join(inp_path, self._get_point_name(point)), inp, delimiter=',')
                np.savetxt(os.path.join(tar_path, self._get_point_name(point)), tar, delimiter=',')
                
                if self.save_img:
                    img_path = os.path.join(path_to, self.image_folder_name)

                    fig, ax = plot(inp[0])
                    ax.scatter(tar, inp[:, tar], c='r', marker='o', s=30)

                    fig.savefig(os.path.join(img_path, self._get_point_name(point)))
                    plt.close(fig)

    def _get_point_name(self, point: Point) -> str:
        return "_".join(map(str, point.info.values()))

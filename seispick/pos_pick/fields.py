import os
import copy
import numpy as np
from torch import sigmoid
from typing import Union
from ..utils.helper import Target
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .datasets.dataset import PositionPickingDataSet
from torch.utils.data import Dataset, DataLoader


class WrapPositionPickingDataSet(PositionPickingDataSet):
    def __init__(self, path: str, transforms: list[callable] = None):
        super(WrapPositionPickingDataSet, self).__init__(path, transforms)

    def __getitem__(self, key):
        inp_path, tar_path = self._inp_paths[key], self._tar_path[key]

        inp_tar = self._load_inp_tar(inp_path, tar_path)

        title = inp_path[inp_path.rfind('/')+1:]

        if self.transforms is not None:
            transformed = self.transforms(image=inp_tar['input'], mask=inp_tar['target'])
            inp_tar['input'], inp_tar['target'] = transformed['image'], transformed['mask']

        inp_tar['title'] = title
        return inp_tar

class DataSetFromLine(Dataset):
    def __init__(self,
                 inp_tar: dict[np.ndarray, Target],
                 field: list[tuple[int, int]],
                 transforms: list[callable] = None):
        
        self.inp_tar = inp_tar
        self.transforms = transforms
        self.field = field
        
    def __getitem__(self, key):
        inp_tar = copy.deepcopy(self.inp_tar)
        x_i, y_i = self.field[key]
        
        inp_tar['target'].sx = self.inp_tar['target'].sx + x_i
        inp_tar['target'].sy = self.inp_tar['target'].sy + y_i
        
        if self.transforms is not None:
            transformed = self.transforms(image=inp_tar['input'], mask=inp_tar['target'])
            inp_tar['input'], inp_tar['target'] = transformed['image'], transformed['mask']
            
        return inp_tar
    
    def __len__(self):
        return len(self.field)

class GetFields:
    def __init__(self,
                 model,
                 data_set: WrapPositionPickingDataSet,
                 x_limits: Union[tuple, int] = 40,
                 y_limits: Union[tuple, int] = 40,
                 delta: Union[tuple, int] = 1,
                 with_pics: bool = True,
                 transforms: list[callable] = None,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 merge: bool=True):
        
        self._pics = 'pics'
        self._output = 'output'
        
        self.data_set = data_set
        self.transforms = transforms
        
        self.x_limits = self._get_limits(x_limits)  
        self.y_limits = self._get_limits(y_limits) 
        self.delta = self._get_limits(delta, is_delta=True)
        self.with_pics = with_pics
        
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.merge = merge
        
        self.field = self._get_field()
        
        
    def save_fields(self, path: str):
        if not os.path.isdir(path) or not isinstance(path, str):
            raise ValueError(f'Wrong path: {path}')
            
        self._make_dirs(path)
        
        x_shape = (self.x_limits[1] - self.x_limits[0] - 1) // self.delta[0] + 1
        y_shape = (self.y_limits[1] - self.y_limits[0] - 1) // self.delta[1] + 1

        if self.merge:
            merged_output = np.zeros(shape=(x_shape, y_shape))
        
        for inp_tar in self.data_set:
            ind = 0
            output = np.ones(shape=(x_shape, y_shape))
            
            line_dataset = DataSetFromLine(inp_tar,
                                           self.field,
                                           self.transforms)
            
            line_dataloader = DataLoader(line_dataset, 
                                         batch_size=self.batch_size, 
                                         num_workers=self.num_workers)
            
            for batch in line_dataloader:
                X_batch, y_batch, title_batch = batch["input"], batch["target"], batch['title']
                y_pred = self.model(X_batch)
                
                for y in y_pred:
                    output[y_shape - ind // x_shape - 1, ind % x_shape] = sigmoid(y)
                    ind += 1

            
            np.savetxt(os.path.join(path, self._output, title_batch[0]), output, delimiter=',')
            self._save_pic(output, path, title_batch[0])

            if self.merge:
                merged_output += output
        
        if self.merge:
            merged_output /= len(self.data_set)
            self.all = np.unravel_index(np.argmax(merged_output), merged_output.shape)
            np.savetxt(os.path.join(path,  self._output, 'All'), merged_output, delimiter=',')
            self._save_pic(merged_output, path, 'All')

                
        
    def _get_field(self):
        x = np.arange(*self.x_limits, self.delta[0])
        y = np.arange(*self.y_limits, self.delta[1])
        return [(xi, yi) for yi in y for xi in x]
        
    def _get_limits(self, limits:  Union[tuple, int], is_delta: bool = False) -> tuple[int, int]:
        if isinstance(limits, int):
            limits = (-limits, limits + 1) if not is_delta else (limits, limits)
        else:
            limits = (limits[0], limits[1]+1) if not is_delta else limits 
        return limits
    
    def _make_dirs(self, path: str):
        pics_path = os.path.join(path, self._pics)
        output_path = os.path.join(path, self._output)
        
        if not os.path.exists(pics_path) and self.with_pics:
            os.makedirs(pics_path)
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    
    def __len__(self):
        return len(self.field)

    def _save_pic(self, output: np.ndarray, path: str, title: str):

        dx, dy = self.delta

        y, x = np.mgrid[slice(self.y_limits[0], self.y_limits[1] + dy, dy),
                        slice(self.x_limits[0], self.x_limits[1] + dx, dx)]

        z = output

        levels = MaxNLocator(nbins=10).tick_values(z.min(), z.max())


        cmap = plt.get_cmap('PiYG')
        norm = mc.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(10, 10))

        im = ax0.pcolormesh(x, y, z, cmap=cmap, norm=norm)
        fig.colorbar(im, ax=ax0)
        ax0.set_title('pcolormesh with levels')


        cf = ax1.contourf(x[:-1, :-1] + dx/2.,
                        y[:-1, :-1] + dy/2., z, levels=levels,
                        cmap=cmap)
        fig.colorbar(cf, ax=ax1)
        ax1.set_title(title)
        fig.tight_layout()
        fig.savefig(os.path.join(path, self._pics, title))
        plt.close(fig)

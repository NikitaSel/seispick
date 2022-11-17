import os
import numpy as np
from typing import Optional
from .readsgy import Data
from .utils.utils import make_path, clear_folder


class PropertiesMixin:
    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        if not os.path.isdir(path):
            raise NotADirectoryError(f'wrong path {path}')
        self._path = path

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        if p > 1 or p < 0:
            raise ValueError(f'wrong p: {p}. p has to be 0 <= p <= 1')
        self._p = p
    

class BaseSaveData(PropertiesMixin):
    def __init__(self,
                 path_to: str,
                 data: Data, 
                 transforms: Optional[list[callable]] = None,
                 p: int = 0.15) -> None:

        self.path = path_to
        self.data = data
        self.transforms = transforms
        self.p = p

        self._init_folder_names()

    def save(self):
        raise NotImplementedError
    
    def _init_folder_names(self, inputs_folder_name: Optional[str] = None, 
                                 targets_folder_name: Optional[str] = None) -> None:
        if inputs_folder_name is None:
            if not hasattr(self, 'inputs_folder_name'):
                self.inputs_folder_name = 'inputs'
        else:
             self.inputs_folder_name = inputs_folder_name

        if targets_folder_name is None:
            if not hasattr(self, 'targets_folder_name'):
                self.targets_folder_name = 'targets'
        else:
             self.targets_folder_name = targets_folder_name

    def _make_paths(self) -> None:
        train_path = make_path(self.path, 'train')
        test_path = make_path(self.path, 'test')

        for path in (train_path, test_path):
            _ = make_path(path, self.inputs_folder_name)
            _ = make_path(path, self.targets_folder_name)

    def get_train_or_test_path(self) -> str:
        if (np.random.random() < self.p):
            return os.path.join(self.path, 'test')

        return os.path.join(self.path, 'train')

    def clear_data(self) -> None:
        clear_folder(self.path)



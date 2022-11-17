import os
from typing import Optional
from torch.utils.data import Dataset
from .utils.utils import get_filepaths_from_dir

class PickingDataSet(Dataset):
    def __init__(self, 
                 path: str, 
                 transforms: Optional[list[callable]] = None):

        self.path = path
        self.transforms = transforms
        self._init_folder_names()

        get_paths = lambda fold_name: self._get_file_paths(os.path.join(path, fold_name))
        self._inp_paths, self._tar_path = map(get_paths, (self.inputs_folder_name, 
                                                          self.targets_folder_name))
    
    def __getitem__(self, key):
        inp_path, tar_path = self._inp_paths[key], self._tar_path[key]
        inp_tar = self._load_inp_tar(inp_path, tar_path)

        if self.transforms is not None:
            transformed = self.transforms(image=inp_tar['input'], mask=inp_tar['target'])
            inp_tar['input'], inp_tar['target'] = transformed['image'], transformed['mask']

        return inp_tar

    def __len__(self):
        return len(self._inp_paths)

    def _get_file_paths(self, path: str) -> list[str]:
        return get_filepaths_from_dir(path)

    def _load_inp_tar(self, inp_path: str, tar_path: str) -> dict:
        inp_tar = {'input': None, 'target': None}
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
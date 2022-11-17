import os
import json
import numpy as np
from functools import partial
from ...utils.helper import Target


def get_file_paths(path: str) -> list[str]:
        paths = list(
                     map(
                         lambda x: os.path.join(path, x),
                                   sorted(os.listdir(path))
                        )
                    )
        return paths

def load_inp_tar(inp_path: str, tar_path: str) -> dict:
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

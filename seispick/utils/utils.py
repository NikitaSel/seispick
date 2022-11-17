import os
import numpy as np


def calculate_3D_offsets(sx: np.ndarray, sy: np.ndarray,
                         gx: np.ndarray, gy: np.ndarray, 
                         rge: np.ndarray, sd: np.ndarray):
    return np.sqrt((gx - sx)**2 + (gy - sy)**2 + (np.abs(rge) - np.abs(sd))**2)


def offsets_to_targets(offsets: np.ndarray, 
                       tr_int: np.ndarray, drt: np.ndarray, 
                       V: int):
    return (np.abs(offsets) / V / tr_int * 1e+6 + drt \
            / tr_int * 1e+3).astype(int) - 1

def make_path(path: str, rest_path: str) -> str:
    if not os.path.isdir(path):
        raise NotADirectoryError(f'wrong path {path}')

    full_path = os.path.join(path, rest_path)

    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path

def get_filepaths_from_dir(path: str) -> list[str]:
    if not os.path.isdir(path):
        raise NotADirectoryError(f'wrong path {path}')

    files = list(
                 map(lambda x: os.path.join(path, x), 
                               os.listdir(path)
                    )
                )
    return files

def clear_folder(path: str) -> None:
    filepaths = get_filepaths_from_dir(path)

    for filepath in filepaths:
        if os.path.isfile(filepath):
            os.remove(filepath)

        if os.path.isdir(filepath):
            if not len(os.listdir(filepath)):
                clear_folder(filepath)
            os.rmdir(filepath)
    
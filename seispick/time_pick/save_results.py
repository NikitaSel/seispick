# import sys
# sys.path.append('/home/seleznev.ns/gradwork_ordered')

import os
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from seispick.plotsgy import plot
from ..utils.utils import make_path
from ..readsgy import Data
from .datasets.dataset import ResultDataset


class SaveResults:
    def __init__(self, 
                 path_to: str,
                 dataset: ResultDataset,
                 model: nn.Module,
                 save_img: bool = False) -> None:

        self.score = 0
        self.batch_size = 20
        self.num_workers = 8

        self.path = path_to
        self.dataset = dataset
        self.model = model
        self.save_img = save_img
        self._init_folder_names()

    def save(self) -> None:
        self._make_paths()

        dataloader = DataLoader(self.dataset, self.batch_size, self.num_workers)

        tmp = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                X_batch, y_batch, info = batch['input'], batch['target'], batch['info']
                y_pred = self.model(X_batch)
                y_pred = y_pred.sum(axis=(1, 2)).argmax(dim=1)[:, None]
                y_batch = y_batch.argmax(dim=1)[:, None]

                for i, (X, y_true, y) in enumerate(zip(X_batch.numpy()[:, 0, 16, :], 
                                                       y_batch.numpy(), 
                                                       y_pred.numpy())):
                    if np.abs((y - y_true)[0]) < 10:
                        tmp.append((y - y_true)[0])

                    result_path = os.path.join(self.path, self.result_folder_name)
                    np.savetxt(os.path.join(result_path, info[i]), y, delimiter=',')

                    if self.save_img:
                        picks_path = os.path.join(self.path, self.pics_folder_name)

                        fig, ax = plot(X)
                        ax.scatter(y, X[y], c='r', marker='o', s=30)
                        ax.scatter(y_true, X[y_true], c='g', marker='o', s=30)

                        fig.savefig(os.path.join(picks_path, info[i]))
                        plt.close(fig)

                self.score += ((y_pred - y_batch) ** 2).sum().to(torch.float)
            self.score /= len(self.dataloader.dataset)
            self.score = self.score.sqrt()

    def _init_folder_names(self, result_folder_name: Optional[str] = None, 
                                 pics_folder_name: Optional[str] = None) -> None:
        if result_folder_name is None:
            if not hasattr(self, 'result_folder_name'):
                self.result_folder_name = 'results'
        else:
             self.result_folder_name = result_folder_name

        if pics_folder_name is None:
            if not hasattr(self, 'pics_folder_name'):
                self.pics_folder_name = 'pics'
        else:
             self.pics_folder_name = pics_folder_name

    def _make_paths(self) -> None:
        _ = make_path(self.path, self.result_folder_name)   
        _ =  make_path(self.path, self.pics_folder_name)

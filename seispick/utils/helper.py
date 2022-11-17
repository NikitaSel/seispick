import numpy as np


class PltSettings:
    def __init__(self,
                 figsize=None,
                 xlabel=None,
                 ylabel=None,
                 title=None,
                 grid=None,
                 vmin=None,
                 vmax=None,
                 cmap=None):

        self.figsize = figsize
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.grid = grid
        self.vmin = vmin
        self.vmax = vmax
        self.cmap = cmap


class Target:
    def __init__(self,
                 sx: np.ndarray, sy: np.ndarray,
                 gx: np.ndarray, gy: np.ndarray,
                 rge: np.ndarray, sd: np.ndarray,
                 tr_int: np.ndarray, drt: np.ndarray):

        self.sx = sx
        self.sy = sy
        self.gx = gx
        self.gy = gy
        self.rge = rge
        self.sd = sd
        self.tr_int = tr_int
        self.drt = drt

    def __getitem__(self, key):
        new_tar = Target(self.sx[key], self.sy[key],
                        self.gx[key], self.gy[key],
                        self.rge[key], self.sd[key],
                        self.tr_int[key], self.drt[key])
        return new_tar

    def __len__(self):
        return len(self.sx)


class SetForCriteria:
    def __init__(self,
                 speed=1442,
                 speed_error=5,
                 dist_between_source=60,
                 freq=500):

        self.speed = speed
        self.speed_error = speed_error
        self.dist_between_source = dist_between_source
        self.freq = freq

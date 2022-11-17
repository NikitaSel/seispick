import numpy as np
from ...utils.helper import Target
from ...utils.utils import calculate_3D_offsets, offsets_to_targets


def custom_crop_by_targets(traces: np.ndarray, 
                           tars: Target, 
                           V: int, width: int = 64):
    shape = traces.shape

    if shape[1] < width:
        raise ValueError(f'Width must be less than shape of cols in traces: {shape[1]}')

    offsets = calculate_3D_offsets(tars.sx, tars.sy, 
                                   tars.gx, tars.gy, 
                                   tars.rge, tars.sd)

    targets = offsets_to_targets(offsets, tars.tr_int, tars.drt, V).astype(int)

    cropped_traces = np.zeros(shape=(traces.shape[0], width))

    left, right = width - width // 2, width // 2

    for i, row in enumerate(traces):

        if targets[i] < left:
            start = left - targets[i] - 1
            cropped_traces[i, start:] = traces[i, :targets[i] + right + 1]
        elif shape[1] < right + targets[i] + 1:
            end = right + targets[i] + 1 - shape[1]
            cropped_traces[i, :shape[1] - end] = traces[i, targets - left + 1:]
        else:
            cropped_traces[i, :] = traces[i, targets[i] - left + 1:targets[i] + right + 1]

    return cropped_traces


def random_uniform_ring(center=np.array([0, 0]),
                        R=1,
                        r=0,
                        nsamples=1):
    nd = len(center)
    x = np.random.normal(size=(nsamples, nd))
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    u = np.random.uniform(size=(nsamples))
    sc = (u * (R**nd - r**nd) + r**nd)**(1 / nd)
    return x * sc[:, np.newaxis] + center


def random_shift_target(tars: Target, min_max_radius: tuple[int, int]): 
    new_gx_gy = random_uniform_ring(center=np.array([tars.gx[0], tars.gy[0]]),
                                    R=min_max_radius[1],
                                    r=min_max_radius[0]) * np.ones(shape=(tars.gx.shape[0], 1))

    tars.gx, tars.gy = new_gx_gy[:, 0], new_gx_gy[:, 1]
    return tars

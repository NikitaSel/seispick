import numpy as np
from ...utils.helper import Target, SetForCriteria
from ...utils.utils import calculate_3D_offsets, offsets_to_targets

def _mess_up_targets(targets: np.ndarray, max_dist=20):
    targets = np.array(targets)
    targets_new = np.random.randint(-max_dist, high=max_dist, size=(targets.shape[0], 1))
    targets_new = targets - targets_new
    is_negative = targets_new < 0
    targets_new[is_negative] = targets[is_negative]
    return targets_new - targets, targets_new

def crop(traces: np.ndarray,
         tars: Target, 
         V: int,
         width=80, mess=False):

        if (traces is None) and (tars is None):
            return (None, None)
                
        tr_h, tr_w = traces.shape

        offsets = calculate_3D_offsets(tars.sx, tars.sy, 
                                       tars.gx, tars.gy, 
                                       tars.rge, tars.sd)

        targets = offsets_to_targets(offsets, tars.tr_int, tars.drt, V)
        targets = targets[:, np.newaxis]
        tar_h, tar_w = targets.shape
        
        if tar_w != 1 or tar_h != tr_h:
            raise ValueError("The targets shape is incompatible with the traces." +  
                             "The targets shape should be [H, 1], the traces shape should be [H, W]")
            
        if mess:
            diff, targets = _mess_up_targets(targets, max_dist = width//4 + 1)

        left = np.zeros(shape=(tar_h, tar_w), dtype=int) + (width - width // 2 - 1)
        right = np.zeros(shape=(tar_h, tar_w), dtype=int) + width // 2

        mask_l = targets < left
        right[mask_l] += (left[mask_l] - targets[mask_l]) 

        mask_r = (tr_w - 1 - targets) < right
        left[mask_r] += (right[mask_r] - (tr_w - 1 - targets[mask_r]))

        mask = np.zeros(shape=(tr_h, tr_w), dtype=int) + np.arange(tr_w)[np.newaxis, :] - targets

        mask[(mask <= right) & (mask >= -left)] = 1
        mask[(mask > right) | (mask < -left)] = 0
        mask = mask.astype(bool)
        
        if width == 0:
            tar = []
        else:
            tar = mask.sum(axis=1) - mask.sum(axis=1) // 2 - 1
            mask_tar = (targets - left).reshape(-1) < 0
            tar[mask_tar] = targets[mask_tar].reshape(-1)
            
        return traces[mask].reshape(tr_h, -1), tar[:, np.newaxis] if not mess else tar[:, np.newaxis] - diff 
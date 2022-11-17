import numpy as np
from ...utils.helper import Target
from ...utils.utils import calculate_3D_offsets

def criteria_for_cut(traces: np.ndarray,
                     tars: Target,
                     V: int):
    right_shift = 100
        
    offsets = calculate_3D_offsets(tars.sx, tars.sy, 
                                   tars.gx, tars.gy, 
                                   tars.rge, tars.sd)
    
    mask = np.where(offsets <= V * (traces.shape[1] - right_shift) / 1e6 * tars.tr_int)[0]
    
    return (traces[mask], tars[mask]) if len(mask) else (None, None)

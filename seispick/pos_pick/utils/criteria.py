import numpy as np
from ...utils.helper import SetForCriteria, Target
from ...utils.utils import offsets_to_targets, calculate_3D_offsets


def t_error_from_speed_error(offset: np.ndarray,
                             settings: SetForCriteria = None):
    if settings is None:
        settings = SetForCriteria()

    t_error = offset / settings.speed**2 * settings.speed_error * settings.freq
    return t_error


# def offsets2t(offsets, settings: SetForCriteria = None):

#     if settings is None:
#         settings = SetForCriteria()

#     effective_depth = np.min(offsets)
#     t = (settings.freq * settings.dist_between_source / settings.speed \
#          * np.sqrt(1 - (effective_depth / offsets)**2))
#     return t


def offsets2t(offsets, tr_int, drt, settings: SetForCriteria=None):
    
    if settings is None:
        settings = SetForCriteria()
    
    return offsets_to_targets(offsets, tr_int, drt, settings.speed)


# def criteria_for_cut(traces: np.ndarray,
#                      tars: Target,
#                      window_size: int,
#                      settings: None,
#                      error_ratio: int = 2):

#     offsets = calculate_3D_offsets(tars.sx, tars.sy, 
#                                    tars.gx, tars.gy, 
#                                    tars.rge, tars.sd)
    
#     arg_min_off = np.argmin(offsets)

#     t = offsets2t(offsets, settings=None)

#     left, right = len(t), 0
#     for i in range(window_size, len(t)):

#         delta_t_from_speed_error = t_error_from_speed_error(offsets[i], settings)
#         delta_t_from_offsets = np.abs(t[i] - t[i-window_size])

#         if delta_t_from_offsets >= delta_t_from_speed_error * error_ratio or i - window_size < arg_min_off < i:
#             if i - window_size < left:
#                 left = i - window_size
#             if i > right:
#                 right = i

#     return traces[left:right], tars[left:right] if right > left else (None, None)


def criteria_for_cut(traces: np.ndarray,
                        tars: Target,
                        window_size: int,
                        settings: SetForCriteria=None,
                        error_ratio: int = 2):

    offsets = calculate_3D_offsets(tars.sx, tars.sy, 
                                   tars.gx, tars.gy, 
                                   tars.rge, tars.sd)

    t =  offsets2t(offsets, tars.tr_int, tars.drt, settings)

    delta_t_from_speed_error = t_error_from_speed_error(offsets, settings)
    
    delta_t_from_offsets = np.zeros_like(delta_t_from_speed_error)
    delta_t_from_offsets[1:] = np.abs(np.diff(t))
    delta_t_from_offsets[0] = delta_t_from_offsets[1]

    left, right = 0, 0
    max_left, max_right = 0, 0 

    mask = np.zeros_like(t, dtype=bool)

    for i in range(len(t)):
        if delta_t_from_offsets[i] >= delta_t_from_speed_error[i] * error_ratio:
            right = i

            if right - left >= window_size:
                if right - left > max_right - max_left:
                    max_left, max_right = left, right
        else:
            left = i

    mask[max_left:max_right+1] = True

    return (traces[mask], tars[mask]) if max_right else (None, None)
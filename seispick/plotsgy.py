from .readsgy import Station, Line, Point
from .utils.helper import PltSettings

import matplotlib.pyplot as plt
import numpy as np
import copy

    
def _is_correct_input(x):
    is_class_point = False
    is_1D = False
            
    if isinstance(x, (Point, Line, Station)):
        is_class_point = True
        
    if isinstance(x, Point) or (isinstance(x, np.ndarray) and len(x.shape) == 1):
        is_1D = True
    elif  isinstance(x, (Line, Station)) or (isinstance(x, np.ndarray) and len(x.shape) == 2):
        is_1D = False
    else:
        raise ValueError('Such input is not supported')
        
    if isinstance(x, np.ndarray) and x.shape[0] == 1:
        print('Warning: x.shape[0]== 1, seems to be a single signal, better to reduce dimension')
    return is_class_point, is_1D
    
def _set_default_settings(x,
                          plt_settings, 
                          is_class_point=True, 
                          is_1D=True):
    
    default_setings = copy.deepcopy(plt_settings)
    
    if plt_settings.title is None and is_class_point:
        text = ''
        for (key, value) in x.info.items():
            if text:
                text = ', '.join([text, f'{key}: {value}'])
            else:
                text += f'{key}: {value}'
                
        default_setings.title = text
        
    if plt_settings.xlabel is None:
        default_setings.xlabel = 'time[k]'
        
    if plt_settings.ylabel is None:
        default_setings.ylabel = 'Amplitude' if is_1D else 'Traces'  
        
    if plt_settings.grid is None:
        default_setings.grid = True if is_1D else False
        
    if plt_settings.figsize is None:
        default_setings.figsize = (15, 5) if is_1D else (15, 15)
        
    if not is_1D and (plt_settings.vmin is None):
        default_setings.vmin = -0.01
        
    if not is_1D and (plt_settings.vmax is None):
        default_setings.vmax = 0.01
        
    if not is_1D and (plt_settings.cmap is None):
        default_setings.cmap = plt.cm.seismic
        
    return default_setings

def _plot_helper(x, 
                 plt_settings, 
                 is_class_point=True, 
                 is_1D=True):
    
    default_setings = _set_default_settings(x,
                                            plt_settings, 
                                            is_class_point=is_class_point, 
                                            is_1D=is_1D) 
    
    if is_1D:
        if isinstance(x, Point):
            trace = x.data['traces'][0, :]  
            width = x.signal_shape[1]
            time = np.arange(0, width, 1)
        else:
            trace = x
            time = np.arange(0, len(x), 1)
        return time, trace, default_setings
    
    if not is_1D:
        if isinstance(x, (Line, Station)):
            traces = x.stack_points()['traces']
        else:
            traces = x
        return traces, default_setings
    
def _pplot(plt_settings, is_1D, *args):
    if len(args) == 2:
        if not is_1D:
            raise ValueError('Number of arguments (two) is incompatible with the parameter is_1D=False')
        x, y = args
    elif len(args) == 1:
        if is_1D:
            raise ValueError('Number of arguments (one) is incompatible with the parameter is_1D=True')
        matr = args[0].T
        if len(matr.shape) != 2:
            raise ValueError('Have to be a 2D matrix')
    else:
        raise ValueError(f'Number of arguments ({len(args)}) is not supported')
    
    fig, ax = plt.subplots(figsize=plt_settings.figsize)
    if is_1D:
        ax.plot(x, y)
    else:
        ax.imshow(matr, 
                    vmin=plt_settings.vmin, 
                    vmax=plt_settings.vmax, 
                    cmap=plt_settings.cmap)
    
    ax.set_xlabel(plt_settings.xlabel)
    ax.set_ylabel(plt_settings.ylabel)
    ax.set_title(plt_settings.title)
    
    if plt_settings.grid:
        ax.grid()
    
    # plt.show()
    
    return fig, ax
     
def plot(x, plt_settings: PltSettings=None):
    plt_settings = plt_settings if plt_settings is not None else PltSettings()
    
    is_class_point, is_1D = _is_correct_input(x)
    
    *data, default_setings = _plot_helper(x, 
                                          plt_settings, 
                                          is_class_point, 
                                          is_1D)
    
    fig, ax = _pplot(default_setings, is_1D, *data)
    return fig, ax

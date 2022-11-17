import torch
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import (
    DualTransform,
    ImageOnlyTransform,
    BasicTransform)
from .f import transform_target, transform_trace

class MaskOnlyTransform(BasicTransform):
    """Transform applied to mask only."""

    @property
    def targets(self) -> dict[str, callable]:
        return {"mask": self.apply}

class FlattenTargets(MaskOnlyTransform):
    def __init__(self, 
                 epsilon: float,
                 always_apply: bool = False, 
                 p: float = 1.0):
            
        super(FlattenTargets, self).__init__(always_apply, p)
        self.epsilon = epsilon

    def apply(self, img: np.ndarray, cols = None, **params) -> np.ndarray:
        assert cols != 0
        return transform_target(img, cols, self.epsilon)

class InputsToMatrix(ImageOnlyTransform):
    def __init__(self, 
                 size: int,
                 always_apply: bool = False, 
                 p: float = 1.0):

         super(InputsToMatrix, self).__init__(always_apply, p)
         self.size = size

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return transform_trace(img, self.size)

class ToTensor(DualTransform):
    def __init__(self, 
                 always_apply: bool = False, 
                 p: float = 1):
        
        super(ToTensor, self).__init__(always_apply, p)
        
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return torch.from_numpy(img)

class ToFloat(DualTransform):
    def __init__(self, 
                    always_apply: bool = False, 
                    p: float = 1): 
                    
        super(ToFloat, self).__init__(always_apply, p)
            
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return img.to(torch.float)

import torch
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import (
    DualTransform,
    ImageOnlyTransform,
    BasicTransform)
from ...utils.helper import Target
from ..utils.utils import custom_crop_by_targets, random_shift_target


class CustomRandCrop(DualTransform):
    def __init__(self, 
                 width: int = 64,
                 always_apply: bool = False, 
                 p: float = 1):
        
        super(CustomRandCrop, self).__init__(always_apply, p)
        self.width = width
        
    def apply(self, img: np.ndarray, mask: np.ndarray = None, **params) -> np.ndarray:
        assert mask is not None
        return img[mask, :]
    
    def apply_to_mask(self, img: Target, mask: np.ndarray = None, **params) -> np.ndarray:
        assert mask is not None
        return img[mask]
    
    def get_params_dependent_on_targets(self, params: dict[str, any]) -> dict[str, any]:
        img = params["image"]
        shape = img.shape
        
        if shape[0] < self.width:
            raise ValueError(f'Width must be less than number of rows in image: {shape[0]}')
        
        rnd_left_ind = np.random.randint(0, shape[0] - self.width + 1)
        
        mask = np.zeros(shape=shape[0], dtype=bool)
        mask[rnd_left_ind:rnd_left_ind+self.width] = True
        
        return {'mask': mask}
        
    @property
    def targets_as_params(self) -> list[str]:
        return ["image"]


class CustomCropByTargets(ImageOnlyTransform):
    def __init__(self,
                 V: int,
                 width: int = 64,
                 always_apply: bool = False,
                 p: float = 1):

        super(CustomCropByTargets, self).__init__(always_apply, p)
        self.V = V
        self.width = width

    def apply(self, img: np.ndarray, cropped_img: np.ndarray = None, **params) -> np.ndarray:
        assert cropped_img is not None
        return cropped_img

    def get_params_dependent_on_targets(self, params: dict[str, any]) -> dict[str, any]:
        img = params['image']
        tar = params['mask']

        try:
            cropped_img = custom_crop_by_targets(img, tar, self.V, self.width)
        except ValueError:
            print(f'Width must be less than shape of cols in image: {img.shape[1]}')
            raise

        return {'cropped_img': cropped_img}

    @property
    def targets_as_params(self) -> list[str]:
        return ["image", "mask"]


class CustomRandomShiftLine(BasicTransform):
    def __init__(self,
                 min_max_radius: tuple[int, int] = (10, 20),
                 always_apply: bool = False,
                 p: float = 0.5):

        super(CustomRandomShiftLine, self).__init__(always_apply, p)
        self.min_max_radius = min_max_radius

    def apply(self, img: Target, tar: Target = None, **params) -> np.ndarray:
        return tar

    def get_params_dependent_on_targets(self, params: dict[str, any]) -> dict[str, any]:
        tar = params['mask']

        if self.min_max_radius[0] > self.min_max_radius[1]:
            raise ValueError(f'The inner radius must be smaller than the outer one in min_max_radius={self.min_max_radius}')

        tar = random_shift_target(tar, min_max_radius=self.min_max_radius)
        tar.is_correct = False

        return {'tar': tar}

    @property
    def targets_as_params(self) -> list[str]:
        return ["mask"]

    @property
    def targets(self) -> dict[str, callable]:
        return {"mask": self.apply}


class CustomChangeTargets(BasicTransform):
    def __init__(self,
                 always_apply: bool = True,
                 p: float = 1):
        super(CustomChangeTargets, self).__init__(always_apply, p)

    def apply(self, img: Target, **params) -> np.ndarray:
        return np.array([0]) if hasattr(img, 'is_correct') else np.array([1])

    @property
    def targets(self) -> dict[str, callable]:
        return {"mask": self.apply}

class CustomAddDim(ImageOnlyTransform):
    def __init__(self, 
                 always_apply: bool = False, 
                 p: float = 1):
        
        super(CustomAddDim, self).__init__(always_apply, p)
        
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return img[np.newaxis, :]

class CustomToTensor(DualTransform):
    def __init__(self, 
                 always_apply: bool = False, 
                 p: float = 1):
        
        super(CustomToTensor, self).__init__(always_apply, p)
        
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return torch.from_numpy(img)

class CustomToFloat(DualTransform):
    def __init__(self, 
                    always_apply: bool = False, 
                    p: float = 1): 
                    
        super(CustomToFloat, self).__init__(always_apply, p)
            
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return img.to(torch.float)

class CustomToFloatNumpy(DualTransform):
    def __init__(self, 
                    always_apply: bool = False, 
                    p: float = 1): 
                    
        super(CustomToFloatNumpy, self).__init__(always_apply, p)
            
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return img.astype('float32')

class CustomCompose(A.Compose):
    def __init__(self, transforms: list[callable]):
        super().__init__(transforms=transforms)
        self.is_check_args = False

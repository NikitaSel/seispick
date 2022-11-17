import numpy as np


class Repr:
    def __repr__(self): return f'{self.__class__.__name__}: {self.__dict__}'


class FunctionWrapperSingle(Repr):
    def __init__(self, function: callable, *args, **kwargs):
        from functools import partial
        self.function = partial(function, *args, **kwargs)

    def __call__(self, inp: np.ndarray): return self.function(inp)


class FunctionWrapperDouble(Repr):
    def __init__(self, function: callable, input: bool = True, target: bool = False, *args, **kwargs):
        from functools import partial
        self.function = partial(function, *args, **kwargs)
        self.input = input
        self.target = target

    def __call__(self, inp: np.ndarray, tar: dict):
        if self.input and not self.target: inp = self.function(inp)
        if self.target and not self.input: tar = self.function(tar)
        if self.target and self.input:
            inp, tar = self.function(inp, tar)
        return inp, tar


class Compose:
    def __init__(self, transforms: list[callable]):
        self.transforms = transforms

    def __repr__(self): return str([transform for transform in self.transforms])


class ComposeDouble(Compose):
    def __call__(self, inp: np.ndarray, target: dict):
        for t in self.transforms:
            inp, target = t(inp, target)
        return inp, target


class ComposeSingle(Compose):
    def __call__(self, inp: np.ndarray):
        for t in self.transforms:
            inp = t(inp)
        return inp

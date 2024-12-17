import functools
import importlib
import sys
from typing import Union
from types import FunctionType, MethodType


def _locate_prev_module(obj):
    if not callable(obj):
        raise ValueError("only support for callable obj")

    if hasattr(obj, "__objclass__"):
        return obj.__objclass__

    if obj.__module__ not in sys.modules:
        importlib.import_module(obj.__module__)

    module = sys.modules[obj.__module__]
    if obj.__qualname__.find(".") == -1:
        return module
    tmp = obj.__qualname__.split(".")[:-1]
    return functools.reduce(getattr, [module] + tmp)


class PatchModule:
    def __init__(self,
                 module: Union[FunctionType, MethodType, str],
                 dst: Union[FunctionType, MethodType, str]):
        if isinstance(module, str):
            module = importlib.import_module(module)

        if isinstance(dst, str):
            if '.' in dst:
                path = dst.split('.')
                dst = functools.reduce(getattr, [module] + path)
            else:
                dst = getattr(module, dst)

        self.module = _locate_prev_module(dst)
        self.dst = dst

    def __call__(self, func: Union[FunctionType, MethodType]):
        @functools.wraps(func)
        def decorated(*args, **kwargs):
            assert self.dst.__name__ not in kwargs, f"{self.dst.__name__}, conflict with patched module args."
            kwargs[self.dst.__name__] = self.dst
            return func(*args, **kwargs)

        setattr(self.module, self.dst.__name__, decorated)
        return func

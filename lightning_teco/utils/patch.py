# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

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

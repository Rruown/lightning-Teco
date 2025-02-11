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


from contextlib import contextmanager
from typing import Any, ContextManager, Generator, Literal, Mapping, Optional, Union

import torch
from typing_extensions import get_args


from lightning_teco.lightning import PrecisionPlugin, rank_zero_info

_PRECISION_INPUT = Literal["32", "32-true",
                           "bf16", "bf16-mixed", "16-mixed",
                           "fp32",
                           "fp32-true",
                           "fp16",
                           "fp16-mixed"]

_AMP_DICT = {
    "fp32": torch.float32,
    "fp32-true": torch.float32,
    "32": torch.float32,
    "32-true": torch.float32,
    "bf16": torch.bfloat16,
    "bf16-mixed": torch.bfloat16,
    "16-mixed": torch.float16,
    "fp16": torch.float16,
    "fp16-mixed": torch.float16,
}


class SDAAPrecisionPlugin(PrecisionPlugin):
    """Plugin that enables mixed precision support on SDAAs.

    Args:
        precision (_PRECISION_INPUT, optional): Precision input. Defaults to "32-true".

    Raises:
        ValueError: Invalid precision value(s).
        NotImplementedError: fp8 / fp16 not available.

    """

    def __init__(
        self,
        precision: _PRECISION_INPUT = "32-true",
        device: str = "sdaa",
    ) -> None:
        supported_precision = get_args(_PRECISION_INPUT)
        if precision not in supported_precision:
            raise ValueError(
                f"`Trainer(accelerator='sdaa', precision={precision!r})` is not supported."
                f" `precision` must be one of: {supported_precision}."
            )
        self.device = device
        self.precision = precision

    def autocast_context_manager(self) -> Union[ContextManager[Any], torch.autocast]:
        """Return Autocast context manager."""
        return torch.autocast(device_type="sdaa", dtype=_AMP_DICT[self.precision], enabled=True)

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Enable autocast context."""
        with self.autocast_context_manager():
            yield


def _replace_layers(module: torch.nn.Module) -> None:
    """Replace layers with Transformer engine equivalent layers.

    Args: torch.nn.Module.
    Return: transformer engine equivalent of torch.nn.Module.
    List of supported modules: https://docs.habana.ai/en/latest/PyTorch/PyTorch_FP8_Training/index.html

    Eg. torch.nn.Linear -> transformer_engine.Linear

    """
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            has_bias = child.bias is not None
            replacement = tengine.Linear(
                child.in_features, child.out_features, bias=has_bias)
            rank_zero_info(
                f"Replacing layer {name} with transformer engine equivalent")
            module.__setattr__(name, replacement)
        else:
            _replace_layers(child)

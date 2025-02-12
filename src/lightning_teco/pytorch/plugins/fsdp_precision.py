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
from typing import Any, ContextManager, Generator, Mapping, Optional, Union

import torch

from typing_extensions import get_args


from lightning_teco.lightning_api import FSDPPrecision

from lightning_teco.pytorch.plugins.precision import _PRECISION_INPUT, SDAAPrecisionPlugin


class _DtypeContextManager:
    """A context manager to change the default tensor type when tensors get created.

    See: :func:`torch.set_default_dtype`

    """

    def __init__(self, dtype: torch.dtype) -> None:
        self._previous_dtype: torch.dtype = torch.get_default_dtype()
        self._new_dtype = dtype

    def __enter__(self) -> None:
        torch.set_default_dtype(self._new_dtype)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch.set_default_dtype(self._previous_dtype)


class SDAAFSDPPrecision(FSDPPrecision, SDAAPrecisionPlugin):
    """Plugin that enables mixed precision support on SDAAs.

    Args:
        precision: to enable ``torch.bfloat16`` (``'bf16-mixed'``).
        device: The device for ``torch.autocast``.

    """

    def __init__(
        self,
        precision: _PRECISION_INPUT,
        device: str = "sdaa",
        recipe: Optional[Union[Mapping[str, Any], "DelayedScaling"]] = None,
        replace_layers: bool = False,
    ) -> None:
        supported_precision = get_args(_PRECISION_INPUT)
        if precision not in supported_precision:
            raise ValueError(
                f"`precision={precision!r}` is not supported." f" `precision` must be one of: {supported_precision}."
            )
        self.precision = precision
        super().__init__(precision)

    def autocast_context_manager(self) -> Union[ContextManager[Any], torch.autocast]:
        """Return Autocast context manager."""
        if "mixed" in self.precision:
            return torch.autocast(device_type="sdaa", dtype=torch.bfloat16, enabled=True)
        return _DtypeContextManager(self._desired_input_dtype)

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Enable autocast context."""
        with self.autocast_context_manager():
            yield

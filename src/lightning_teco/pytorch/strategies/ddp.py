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

import logging
import pytorch_lightning as pl
import torch
from torch import Tensor
from typing import Any, Callable, Dict, List, Optional, Union

from lightning_teco.utils.sdaa_distributed import _sync_sdaa_processes_if_available

from lightning_teco.lightning_api import (CheckpointIO,
                                          ClusterEnvironment,
                                          _WrappingCheckpointIO,
                                          PrecisionPlugin,
                                          DDPStrategy,
                                          ReduceOp,)


from ..plugins.io_plugin import SDAACheckpointIO

log = logging.getLogger(__name__)


def _parse_parallel_devices(parallel_devices: Optional[List[torch.device]]):
    if parallel_devices is None:
        return

    result = []
    for i, device in enumerate(parallel_devices):
        if device.type != "sdaa":
            raise ValueError("SDAA Strategy only support for sdaa device!")
        result.append(torch.device(f"{device.type}:{i}"))
    return result


class SDAADDPStrategy(DDPStrategy):
    """Strategy for distributed training on multiple SDAA devices."""

    strategy_name = "dpp_sdaa"

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        ddp_comm_state: Optional[object] = None,
        ddp_comm_hook: Optional[Callable] = None,
        ddp_comm_wrapper: Optional[Callable] = None,
        model_averaging_period: Optional[int] = None,
        process_group_backend: Optional[str] = "tccl",
        **kwargs: Any,
    ) -> None:

        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
            ddp_comm_state=ddp_comm_state,
            ddp_comm_hook=ddp_comm_hook,
            ddp_comm_wrapper=ddp_comm_wrapper,
            model_averaging_period=model_averaging_period,
            process_group_backend=process_group_backend,
            **kwargs,
        )

        self.parallel_devices = _parse_parallel_devices(self.parallel_devices)

    def _get_process_group_backend(self):
        return "tccl"

    def determine_ddp_device_ids(self) -> None:
        return None

    def reduce(
        self,
        tensor: Union[Tensor, Any],
        group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = "mean",
    ) -> Union[Tensor, Any]:
        if isinstance(tensor, Tensor):
            if tensor.device != self.root_device:
                tensor = tensor.to(self.root_device)
            return _sync_sdaa_processes_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    @property
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:
            self._checkpoint_io = SDAACheckpointIO()
        elif isinstance(self._checkpoint_io, _WrappingCheckpointIO):
            self._checkpoint_io.checkpoint_io = SDAACheckpointIO()

        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: Optional[CheckpointIO]) -> None:
        self._checkpoint_io = io

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            "ddp_sdaa_find_unused_parameters_false",
            cls,
            description="SDAA DDP Strategy with `find_unused_parameters` as False",
            find_unused_parameters=False,
        )
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )

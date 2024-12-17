# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from typing import Any, Callable, Dict, List, Optional

import torch.distributed

import pytorch_lightning as pl
from lightning_lite.plugins import CheckpointIO, ClusterEnvironment
from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException


from lightning_teco.plugins.io_plugin import SDAACheckpointIO
from lightning_teco.utils.imports import _SDAA_AVAILABLE

log = logging.getLogger(__name__)


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

        if not _SDAA_AVAILABLE:
            raise MisconfigurationException(
                "`SDAAParallelStrategy` requires SDAA devices to run")

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

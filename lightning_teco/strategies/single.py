# Copyright The Lightning AI team.
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
import torch
from typing import Any, Callable, Dict, Optional, Union

from lightning_utilities import module_available
from pytorch_lightning.utilities import find_shared_parameters, set_shared_parameters

if module_available("pytorch_lightning"):
    from lightning_lite.plugins import CheckpointIO
    from pytorch_lightning import Trainer
    from pytorch_lightning.accelerators import Accelerator
    from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
    from pytorch_lightning.plugins.precision import PrecisionPlugin
    from pytorch_lightning.strategies.single_device import SingleDeviceStrategy
else:
    raise ModuleNotFoundError(
        "You are missing `lightning` or `pytorch-lightning` package, please install it.")

from lightning_teco.plugins.io_plugin import SDAACheckpointIO

class SingleSDAAStrategy(SingleDeviceStrategy):
    """Strategy for training on single SDAA device."""

    strategy_name = "sdaa_single"

    def __init__(
        self,
        device: Union[torch.device, str, int] = "sdaa",
        accelerator: Optional[Accelerator] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
    ):
        super().__init__(
            accelerator=accelerator,
            device=device,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )

    @property
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:  # type: ignore[has-type]
            self._checkpoint_io = SDAACheckpointIO()
        elif isinstance(self._checkpoint_io, _WrappingCheckpointIO):
            self._checkpoint_io.checkpoint_io = SDAACheckpointIO()

        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: Optional[CheckpointIO]) -> None:
        self._checkpoint_io = io  # type: ignore

    @property
    def is_distributed(self) -> bool:
        return False

    def setup(self, trainer: Trainer) -> None:
        assert self.model, "self.model must be set before find_shared_parameters(self.model)"
        shared_params = find_shared_parameters(self.model)
        self.model_to_device()
        set_shared_parameters(self.model, shared_params)
        super().setup(trainer)

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )

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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from lightning_utilities import module_available

import pytorch_lightning as pl
from lightning_teco.utils.imports import _SDAA_AVAILABLE
from lightning_teco.utils.resources import _parse_sdaa_ids, num_sdaa_devices


from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _SDAA_AVAILABLE:
    import torch_sdaa

class SDAAAccelerator(Accelerator):
    """Accelerator for SDAA devices."""

    def setup_device(self, device: torch.device) -> None:
        """Set up the device.

        Raises:
            MisconfigurationException:
                If the selected device is not SDAA.

        """
        if device.type != "sdaa":
            raise MisconfigurationException(
                f"Device should be SDAA, got {device} instead.")
        if device.index is None:
            device = 0
        torch.sdaa.set_device(device)

    def setup(self, trainer: "pl.Trainer") -> None:
        # TODO refactor input from trainer to local_rank @four4fish
        # clear cache before training
        torch.sdaa.empty_cache()

    def get_device_stats(self, device: Union[torch.device, str, int]) -> Dict[str, Any]:
        """Return a map of the following metrics with their values."""
        return torch.sdaa.memory_stats(device)

    def teardown(self) -> None:
        torch.sdaa.empty_cache()

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[int]:
        """Accelerator device parsing logic."""
        return _parse_sdaa_ids(devices)

    @staticmethod
    def get_parallel_devices(devices: int) -> List[torch.device]:
        """Get parallel devices for the Accelerator."""
        return [torch.device("sdaa", i) for i in devices]

    @staticmethod
    def auto_device_count() -> int:
        """Return the number of SDAA devices when the devices is set to auto."""
        return num_sdaa_devices()

    @staticmethod
    def is_available() -> bool:
        """Return a bool indicating if SDAA is currently available."""
        try:
            return torch.sdaa.is_available()
        except (AttributeError, NameError):
            return False

    @staticmethod
    def get_device_name() -> str:
        """Return the name of the SDAA device."""
        try:
            return torch.sdaa.get_device_name()
        except (AttributeError, NameError):
            return ""

    @classmethod
    def register_accelerators(cls, accelerator_registry: Dict) -> None:
        accelerator_registry.register(
            "sdaa",
            cls,
            description=cls.__class__.__name__,
        )

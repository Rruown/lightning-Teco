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

# Copyright (c) 2023 Teco Labs, Ltd. an  Company
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
import operator
import os
import torch

from lightning_utilities import compare_version

from lightning_teco.accelerator import SDAAAccelerator
# from lightning_teco.datamodule.datamodule import SDAADataModule
# from lightning_teco.plugins.deepspeed_precision import SDAADeepSpeedPrecisionPlugin
# from lightning_teco.plugins.fsdp_precision import SDAAFSDPPrecision
# from lightning_teco.plugins.io_plugin import SDAACheckpointIO
# from lightning_teco.plugins.precision import SDAAPrecisionPlugin
from lightning_teco.profiler import SDAAProfiler
# from lightning_teco.strategies.deepspeed import SDAADeepSpeedStrategy
# from lightning_teco.strategies.fsdp import SDAAFSDPStrategy
from lightning_teco.strategies.ddp import SDAADDPStrategy
# from lightning_teco.strategies.single import SingleSDAAStrategy
from lightning_teco.__about__ import __min_required_version__
from lightning_teco.utils.imports import _SDAA_AVAILABLE
from lightning_teco.register import plugin_register

if compare_version("lightning", operator.lt, __min_required_version__) and compare_version("pytorch_lightning", operator.lt, __min_required_version__):
    raise ImportError(
        "You are missing `lightning` or `pytorch-lightning` package or neither of them is in version 1.8.5+"
    )

if _SDAA_AVAILABLE:
    import torch_sdaa
    SDAA_AVAILABLE: bool = torch_sdaa.backend.is_available()
else:
    SDAA_AVAILABLE = False

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)


plugin_register()

__all__ = [
    "SDAAAccelerator",
    "SDAADDPStrategy",
    "SDAAProfiler",
    # "SDAADeepSpeedStrategy",
    # "SDAAParallelStrategy",
    # "SDAAFSDPStrategy",
    # "SingleSDAAStrategy",
    # "SDAAPrecisionPlugin",
    # "SDAACheckpointIO",
    # "SDAADataModule",
    # "SDAAProfiler",
    # "SDAA_AVAILABLE",
    # "SDAADeepSpeedPrecisionPlugin",
    # "SDAAFSDPPrecision",
]

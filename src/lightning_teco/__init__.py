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

import torch
from lightning_teco.pytorch.accelerator import SDAAAccelerator
from lightning_teco.pytorch.plugins.fsdp_precision import SDAAFSDPPrecision
from lightning_teco.pytorch.plugins.deepspeed_precision import SDAADeepSpeedPrecisionPlugin
from lightning_teco.pytorch.plugins.io_plugin import SDAACheckpointIO
from lightning_teco.pytorch.plugins.precision import SDAAPrecisionPlugin
from lightning_teco.pytorch.profiler import SDAAProfiler
from lightning_teco.pytorch.strategies.fsdp import SDAAFSDPStrategy
from lightning_teco.pytorch.strategies.ddp import SDAADDPStrategy
from lightning_teco.pytorch.strategies.single import SingleSDAAStrategy
from lightning_teco.pytorch.strategies.deepspeed import SDAADeepSpeedStrategy
from lightning_teco.utils.imports import check_environment
from lightning_teco.register import *

check_environment()
plugin_register()

# patch for sw-deepseed
import torch_sdaa
if not hasattr(torch.sdaa, 'nvtx'):
    setattr(torch.sdaa, 'nvtx', object())

__all__ = [
    "SDAAAccelerator",
    "SDAADDPStrategy",
    "SDAAProfiler",
    "SDAAFSDPStrategy",
    "SingleSDAAStrategy",
    "SDAAPrecisionPlugin",
    "SDAADeepSpeedPrecisionPlugin",
    "SDAACheckpointIO",
    "SDAAProfiler",
    "SDAAFSDPPrecision",
    "SDAADeepSpeedStrategy",

]

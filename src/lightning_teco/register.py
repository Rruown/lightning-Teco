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

import pytorch_lightning as pl
from pytorch_lightning.accelerators import AcceleratorRegistry
from pytorch_lightning.strategies import StrategyRegistry
from lightning_teco.pytorch.accelerator import SDAAAccelerator
from lightning_teco.pytorch.strategies import SingleSDAAStrategy, SDAADDPStrategy
from lightning_teco.pytorch.profiler import SDAAProfiler
from lightning_teco.utils.patch import PatchModule
from pytorch_lightning.utilities.rank_zero import rank_zero_info


@PatchModule(module='pytorch_lightning.trainer.setup', dst='_init_profiler')
def _path_init_profiler(trainer: "pl.Trainer", profiler, _init_profiler):
    if isinstance(profiler, str) and profiler == 'sdaa_profiler':
        trainer.profiler = SDAAProfiler()
    else:
        _init_profiler(trainer, profiler)


@PatchModule(module='pytorch_lightning.trainer.setup', dst='_log_device_info')
def _patch__log_device_info(trainer: "pl.Trainer", _log_device_info):
    _log_device_info(trainer)

    sdaa_used = isinstance(trainer.accelerator, SDAAAccelerator)
    num_sdaas = trainer.num_devices if isinstance(
        trainer.accelerator, SDAAAccelerator) else 0
    rank_zero_info(f"SDAA available: {sdaa_used}, using: {num_sdaas} SDAAs")


@PatchModule(module='pytorch_lightning.trainer.connectors.accelerator_connector',
             dst='_AcceleratorConnector._choose_strategy')
def _patch_choose_strategy(self, _choose_strategy):
    if self._accelerator_flag == 'sdaa':
        if self._parallel_devices and len(self._parallel_devices) > 1:
            return SDAADDPStrategy.strategy_name
        return SingleSDAAStrategy.strategy_name
    return _choose_strategy(self)


def plugin_register():
    SDAAAccelerator.register_accelerators(AcceleratorRegistry)
    SingleSDAAStrategy.register_strategies(StrategyRegistry)
    SDAADDPStrategy.register_strategies(StrategyRegistry)

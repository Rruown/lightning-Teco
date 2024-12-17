import pytorch_lightning as pl
from pytorch_lightning.accelerators import AcceleratorRegistry
from pytorch_lightning.strategies import StrategyRegistry
from lightning_teco.accelerator import SDAAAccelerator
from lightning_teco.strategies import SingleSDAAStrategy, SDAADDPStrategy
from lightning_teco.profiler import SDAAProfiler
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
             dst='AcceleratorConnector._choose_strategy')
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

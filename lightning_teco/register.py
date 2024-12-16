import pytorch_lightning as pl
from pytorch_lightning.accelerators import AcceleratorRegistry
from pytorch_lightning.strategies import StrategyRegistry
from lightning_teco.accelerator import SDAAAccelerator
from lightning_teco.strategies import SingleSDAAStrategy, SDAADDPStrategy
from lightning_teco.profiler import SDAAProfiler

from pytorch_lightning.trainer import setup


def _patch_profiler():

    old_init_profiler = setup._init_profiler

    def _new_init_Profiler(trainer: "pl.Trainer", profiler):
        if isinstance(profiler, str) and profiler == 'sdaa_profiler':
            trainer.profiler = SDAAProfiler()
        else:
            old_init_profiler(trainer, profiler)

    setup._init_profiler = _new_init_Profiler


def plugin_register():
    SDAAAccelerator.register_accelerators(AcceleratorRegistry)
    SingleSDAAStrategy.register_strategies(StrategyRegistry)
    SDAADDPStrategy.register_strategies(StrategyRegistry)
    _patch_profiler()

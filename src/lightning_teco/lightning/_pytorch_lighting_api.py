import operator
from lightning_utilities import compare_version


if compare_version("pytorch_lightning", operator.gt, "1.8.6"):
    from lightning_fabric.utilities.types import _DEVICE, ReduceOp
    from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment, TorchCheckpointIO
    from lightning_fabric.plugins.environments.torchelastic import TorchElasticEnvironment
    from lightning_fabric.utilities.cloud_io import _atomic_save as atomic_save, get_filesystem
    from lightning_fabric.utilities.distributed import _sync_ddp
    from lightning_fabric.utilities.rank_zero import rank_zero_info, rank_zero_warn
    from pytorch_lightning.plugins.precision import Precision as PrecisionPlugin, FSDPPrecision
    from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
    from lightning_fabric.strategies import _StrategyRegistry
    from lightning_fabric.strategies.fsdp import (
        _move_torchmetrics_to_device,
        _setup_activation_checkpointing,
    )
    from lightning_fabric.utilities.init import _has_meta_device_parameters_or_buffers
else:
    from lightning_lite.utilities.types import _DEVICE, ReduceOp
    from lightning_lite.plugins import CheckpointIO, ClusterEnvironment, TorchCheckpointIO
    from lightning_lite.plugins.environments.torchelastic import TorchElasticEnvironment
    from lightning_lite.utilities.cloud_io import atomic_save, get_filesystem
    from lightning_lite.utilities.distributed import _sync_ddp
    from lightning_lite.utilities.rank_zero import rank_zero_info, rank_zero_warn
    from pytorch_lightning.plugins.precision import PrecisionPlugin, FullyShardedNativeMixedPrecisionPlugin as FSDPPrecision
    from lightning_lite.plugins.collectives.torch_collective import default_pg_timeout
    from lightning_lite.strategies import _StrategyRegistry
    from lightning_lite.strategies.fsdp import (
        FSDPStrategy,
        _move_torchmetrics_to_device,
        _setup_activation_checkpointing,
    )
    from lightning_lite.utilities.init import _has_meta_device_parameters_or_buffers

from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.strategies import DDPStrategy, SingleDeviceStrategy, FSDPStrategy
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
from pytorch_lightning.utilities import move_data_to_device

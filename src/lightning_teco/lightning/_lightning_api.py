import operator
from lightning_utilities import compare_version


if compare_version("lightning", operator.gt, "1.8.6"):
    from lightning.fabric.utilities.types import _DEVICE, ReduceOp
    from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment, TorchCheckpointIO
    from lightning.fabric.utilities.exceptions import MisconfigurationException
    from lightning.fabric.plugins.environments.torchelastic import TorchElasticEnvironment
    from lightning.fabric.utilities.cloud_io import _atomic_save as atomic_save, get_filesystem
    from lightning.fabric.utilities.distributed import _sync_ddp
    from lightning.fabric.utilities.rank_zero import rank_zero_info, rank_zero_warn
    from lightning.pytorch.plugins.precision import Precision as PrecisionPlugin
    from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
    from lightning.fabric.strategies import _StrategyRegistry
    from lightning.fabric.strategies.fsdp import (
        _move_torchmetrics_to_device,
        _setup_activation_checkpointing,
    )
    from lightning.fabric.utilities.init import _has_meta_device_parameters_or_buffers


else:
    from lightning.lite.utilities.types import _DEVICE, ReduceOp
    from lightning.lite.plugins import CheckpointIO, ClusterEnvironment, TorchCheckpointIO
    from lightning.lite.utilities.exceptions import MisconfigurationException
    from lightning.lite.plugins.environments.torchelastic import TorchElasticEnvironment
    from lightning.lite.utilities.cloud_io import atomic_save, get_filesystem
    from lightning.lite.utilities.distributed import _sync_ddp
    from lightning.lite.utilities.rank_zero import rank_zero_info, rank_zero_warn
    from lightning.lite.plugins.collectives.torch_collective import default_pg_timeout
    from lightning.lite.strategies import _StrategyRegistry
    from lightning.lite.strategies.fsdp import (
        _move_torchmetrics_to_device,
        _setup_activation_checkpointing,
    )
    from lightning.lite.utilities.init import _has_meta_device_parameters_or_buffers

from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.plugins.precision import PrecisionPlugin
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy, FSDPStrategy
from lightning.pytorch.plugins.io.wrapper import _WrappingCheckpointIO
from lightning.pytorch.utilities import move_data_to_device
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

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


import logging
from contextlib import contextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Callable, Generator, List, Literal, Optional, Set, Type, Union

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override

from lightning_teco.lightning import (CheckpointIO,
                                      _WrappingCheckpointIO,
                                      ClusterEnvironment,
                                      default_pg_timeout,
                                      _StrategyRegistry,
                                      _move_torchmetrics_to_device,
                                      _setup_activation_checkpointing,
                                      _has_meta_device_parameters_or_buffers,
                                      ReduceOp,
                                      PrecisionPlugin,
                                      FSDPStrategy,
                                      rank_zero_warn
                                      )

from lightning_teco.pytorch.plugins.fsdp_precision import SDAAFSDPPrecision
from lightning_teco.pytorch.plugins.io_plugin import SDAACheckpointIO
from lightning_teco.pytorch.strategies.ddp import SDAADDPStrategy

if TYPE_CHECKING:
    from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    _POLICY = Union[Set[Type[Module]], Callable[[
        Module, bool, int], bool], ModuleWrapPolicy]

    _SHARDING_STRATEGY = Union[ShardingStrategy,
                               Literal["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"]]


log = logging.getLogger(__name__)


class SDAAFSDPStrategy(FSDPStrategy):
    r"""Strategy for Fully Sharded Data Parallel provided by torch.distributed on SDAA.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    """

    strategy_name = "sdaa_fsdp"
    _registered_strategies: List[str] = []

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = SDAACheckpointIO(),
        precision_plugin: Optional[PrecisionPlugin] = SDAAFSDPPrecision(
            "bf16-mixed"),
        process_group_backend: Optional[str] = "tccl",
        timeout: Optional[timedelta] = default_pg_timeout,
        cpu_offload: Union[bool, "CPUOffload", None] = None,
        mixed_precision: Optional["MixedPrecision"] = None,
        auto_wrap_policy: Optional["_POLICY"] = None,
        activation_checkpointing: Optional[Union[Type[Module],
                                                 List[Type[Module]]]] = None,
        activation_checkpointing_policy: Optional["_POLICY"] = None,
        sharding_strategy: "_SHARDING_STRATEGY" = "FULL_SHARD",
        state_dict_type: Literal["full", "sharded"] = "full",
        **kwargs: Any,
    ) -> None:

        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
            process_group_backend=process_group_backend,
            timeout=timeout,
            cpu_offload=cpu_offload,
            mixed_precision=mixed_precision,
            auto_wrap_policy=auto_wrap_policy,
            activation_checkpointing=activation_checkpointing,
            activation_checkpointing_policy=activation_checkpointing_policy,
            sharding_strategy=sharding_strategy,
            state_dict_type=state_dict_type,
            **kwargs,
        )

    @property
    def mixed_precision_config(self) -> Optional["MixedPrecision"]:
        if self.mixed_precision:
            return self.mixed_precision
        plugin = self.precision_plugin
        if isinstance(plugin, SDAAFSDPPrecision):
            return plugin.mixed_precision_config
        return None

    @property
    @override
    def precision_plugin(self) -> SDAAFSDPPrecision:
        plugin = self._precision_plugin
        if plugin is not None:
            return plugin
        return SDAAFSDPPrecision("bf16-mixed")

    @precision_plugin.setter
    @override
    def precision_plugin(self, precision_plugin: Optional[SDAAFSDPPrecision]) -> None:
        if precision_plugin is not None and not isinstance(precision_plugin, SDAAFSDPPrecision):
            raise TypeError(
                f"The FSDP strategy can only work with the `SDAAFSDPPrecision` plugin, found {precision_plugin}"
            )
        self._precision_plugin = precision_plugin


    @property
    def checkpoint_io(self) -> CheckpointIO:
        if self._checkpoint_io is None:
            self._checkpoint_io = SDAACheckpointIO()
        elif isinstance(self._checkpoint_io, _WrappingCheckpointIO):
            self._checkpoint_io.checkpoint_io = SDAACheckpointIO()

        return self._checkpoint_io

    @checkpoint_io.setter
    def checkpoint_io(self, io: Optional[CheckpointIO]) -> None:
        self._checkpoint_io = io

    @property
    @override
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return torch.device("sdaa", torch.sdaa.current_device())

    def _setup_model(self, model: Module) -> Module:
        from torch.distributed.fsdp import FullyShardedDataParallel

        if any(isinstance(mod, FullyShardedDataParallel) for mod in model.modules()):
            if _has_meta_device_parameters_or_buffers(model):
                rank_zero_warn(
                    "The model is already wrapped in `FSDP` but there are still parameters on the meta device."
                )
            if "auto_wrap_policy" in self.kwargs:
                # The user has wrapped their submodules manually, don't apply the auto wrap policy.
                rank_zero_warn(
                    "A FSDP `auto_wrap_policy` is set, but the model is already wrapped. The policy will be ignored."
                )
                del self.kwargs["auto_wrap_policy"]
        else:
            model = FullyShardedDataParallel(
                module=model,
                cpu_offload=self.cpu_offload,
                mixed_precision=self.mixed_precision_config,
                sharding_strategy=self.sharding_strategy,
                device_id=self.root_device,  # Index based device selection is not supported on SDAA
                **self.kwargs,
            )

        _move_torchmetrics_to_device(model, self.root_device)

        # activation checkpointing needs to be set up after wrapping the model
        _setup_activation_checkpointing(
            model, self._activation_checkpointing_kwargs)

        return model

    def setup(self, trainer: "pl.Trainer") -> None:
        self.model_to_device()
        super().setup(trainer)

    @contextmanager
    def model_sharded_context(self) -> Generator[None, None, None]:
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
        from torch.distributed.fsdp.wrap import enable_wrap

        with enable_wrap(
            wrapper_cls=FullyShardedDataParallel,
            cpu_offload=self.cpu_offload,
            mixed_precision=self.mixed_precision_config,
            sharding_strategy=self.sharding_strategy,
            device_id=self.root_device,  # Index based device selection is not supported on SDAA
            **self.kwargs,
        ):
            yield

    def reduce(
        self,
        tensor: Union[Tensor, Any],
        group: Optional[Any] = None,
        reduce_op: Optional[Union[ReduceOp, str]] = "mean",
    ) -> Union[Tensor, Any]:
        # Skipping FSDPStrategy (first in mro) and inheriting from SDAAParallelStrategy.
        return SDAADDPStrategy.reduce(self, tensor, group, reduce_op)

    def _get_process_group_backend(self) -> str:
        return "tccl"

    @classmethod
    def get_registered_strategies(cls) -> List[str]:
        return cls._registered_strategies

    @classmethod
    @override
    def register_strategies(cls, strategy_registry: _StrategyRegistry) -> None:
        if not torch.distributed.is_available():
            return
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description="Fully Sharded Data Parallel (FSDP) training",
        )
        cls._registered_strategies.append(cls.strategy_name)

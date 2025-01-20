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
from lightning_utilities.core.imports import module_available
from torch import Tensor

from pytorch_lightning.utilities.distributed import _sync_ddp
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
from lightning_lite.utilities.types import ReduceOp

from typing import Any, Optional, Union

# Supported ReduceOps: https://docs.teco.ai/en/latest/API_Reference_Guides/HCCL_APIs/C_API.html#tcclredop-t
supported_reduce_ops = {
    "sum": ReduceOp.SUM,
    "min": ReduceOp.MIN,
    "max": ReduceOp.MAX,
}


def _distributed_available() -> bool:
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_backend() == "tccl"
    )


def _is_reduce_op_supported(reduce_op: Union[ReduceOp, str]) -> bool:
    """Function to check if reduce_op is supported with tccl backend."""
    reduce_op = reduce_op.lower() if isinstance(reduce_op, str) else reduce_op
    if reduce_op in ("mean", "avg") or reduce_op == ReduceOp.AVG:
        rank_zero_warn(
            f"{reduce_op} is not supported with TCCL. Going to simulate it")
        return True
    if reduce_op not in supported_reduce_ops and not any(reduce_op is op for op in supported_reduce_ops.values()):
        raise TypeError(
            f"Unsupported ReduceOp {reduce_op}. Supported ops in TCCL are: {', '.join(supported_reduce_ops)}"
        )
    return True


def _sync_sdaa_processes_if_available(
    result: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "sum"
) -> Tensor:
    """Function to reduce a tensor across worker processes during distributed training.

    Args:
        result: The value to sync and reduce (typically tensor or number)
        group: The process group to gather results from. Defaults to all processes (world)
        reduce_op: The reduction operation. Defaults to sum.

    Return:
        reduced value

    """
    assert reduce_op is not None
    if _distributed_available() and _is_reduce_op_supported(reduce_op):
        return _sync_sdaa(result, group=group, reduce_op=reduce_op)
    return result


def _sync_sdaa(result: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "sum") -> Tensor:
    """Reduces a tensor across several distributed processes.

    This operation is performed in-place, meaning the result will be placed back into the input tensor on all processes.

    Args:
        result: The value to sync and reduce (typically tensor or number)
        group: The process group to gather results from. Defaults to all processes (world)
        reduce_op: The reduction operation. Defaults to sum.

    Return:
        The reduced value.

    """
    # Simulate mean using sum
    reduce_op = reduce_op.lower() if isinstance(reduce_op, str) else reduce_op
    op = "sum" if (reduce_op == ReduceOp.AVG or reduce_op in (
        "mean", "avg")) else reduce_op
    result = _sync_ddp(result, group, op)

    if reduce_op == ReduceOp.AVG or reduce_op in ("mean", "avg"):
        # Compute mean from sum
        group = torch.distributed.group.WORLD if group is None else group
        world_size = torch.distributed.get_world_size(group)

        # SDAA doesn't support Long types, forcefully set it to float
        if result.type() in (
            "torch.LongTensor",
            "torch.sdaa.LongTensor",
        ):
            rank_zero_info("Long tensor unsupported on SDAA, casting to float")
            result = result.float()
        return result.div_(world_size)
    return result

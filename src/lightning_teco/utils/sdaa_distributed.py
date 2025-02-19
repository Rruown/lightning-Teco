# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
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

import torch
from torch import Tensor
import torch.distributed


from lightning_teco.lightning_api import _sync_ddp, rank_zero_info, ReduceOp

from typing import Any, Optional, Union


def _distributed_available() -> bool:
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_backend() == "tccl"
    )


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
    if _distributed_available():
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
    reduce_op = reduce_op.lower() if isinstance(reduce_op, str) else reduce_op
    op = "sum" if (reduce_op == ReduceOp.AVG or reduce_op in (
        "mean", "avg")) else reduce_op
    result = _sync_ddp(result, group, op)

    if reduce_op == ReduceOp.AVG or reduce_op in ("mean", "avg"):
        # Compute mean from sum
        group = torch.distributed.group.WORLD if group is None else group
        world_size = torch.distributed.get_world_size(group)
        return result.div_(world_size)
    return result

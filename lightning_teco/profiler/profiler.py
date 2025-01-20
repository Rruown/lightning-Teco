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
import os
import re
from pathlib import Path
from typing import Any, List, Optional, Union

import torch
from lightning_utilities import module_available

from pytorch_lightning.profilers.pytorch import PyTorchProfiler
from pytorch_lightning.trainer.connectors.data_connector import warning_cache
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

from lightning_teco.utils.imports import _KINETO_AVAILABLE

if _KINETO_AVAILABLE:
    from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

log = logging.getLogger(__name__)

_PROFILER = profile


class SDAAProfiler(PyTorchProfiler):
    r"""This profiler subclasses the PyTorch Profiler and lets you inspect the cost of.

    different operators inside your model - both on the CPU and SDAA devices.

    Args:
        dirpath: Directory path for the ``filename``. If ``dirpath`` is ``None`` but ``filename`` is present, the
            ``trainer.log_dir`` (from :class:`~pytorch_lightning.loggers.tensorboard.TensorBoardLogger`)
            will be used.

        filename: If present, filename where the profiler results will be saved instead of printing to stdout.
            The ``.txt`` extension will be used automatically.

        group_by_input_shapes: Include operator input shapes and group calls by shape.

        export_to_chrome: Whether to export the sequence of profiled operators for Chrome.
            It will generate a ``.json`` file which can be read by Chrome.

        row_limit: Limit the number of rows in a table, ``-1`` is a special value that
            removes the limit completely.

        sort_by_key: Attribute used to sort entries. By default
            they are printed in the same order as they were registered.
            Valid keys include: ``cpu_time``, ``cpu_time_total``,
            ``cpu_memory_usage``, ``self_cpu_memory_usage``, ``count``.

        record_module_names: Whether to add module names while recording autograd operation.

        \**profiler_kwargs: Keyword arguments for the PyTorch profiler. This depends on your PyTorch version

    Raises:
        MisconfigurationException:
            If arg ``sort_by_key`` is not present in ``AVAILABLE_SORT_KEYS``.
            If arg ``schedule`` is not a ``Callable``.
            If arg ``schedule`` does not return a ``torch.profiler.ProfilerAction``.

    """

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        group_by_input_shapes: bool = False,
        export_to_chrome: bool = True,
        row_limit: int = 20,
        sort_by_key: Optional[str] = None,
        record_module_names: bool = False,
        **profiler_kwargs: Any,
    ) -> None:
        assert os.environ.get("SDAA_PROFILE", None) in (
            None,
            "profile_api_light",
        ), "`SDAA_PROFILE` should not be set when using `SDAAProfiler`"
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            group_by_input_shapes=group_by_input_shapes,
            export_to_chrome=export_to_chrome,
            row_limit=row_limit,
            sort_by_key=sort_by_key or f"{'sdaa' if profiler_kwargs.get('use_sdaa', False) else 'cpu'}_time_total",
            record_module_names=record_module_names,
            **profiler_kwargs,
        )
        self.profiler: Optional[_PROFILER] = None
        self._profiler_kwargs["activities"] = self.profile_sdaa_activities(
            self._profiler_kwargs.get("activities", None))

    # type: ignore
    def profile_sdaa_activities(self, activities) -> List["ProfilerActivity"]:
        if not _KINETO_AVAILABLE:
            return activities
        activities.append(ProfilerActivity.SDAA)
        return activities

    def stop(self, action_name: str) -> None:
        if action_name in self._recording_map:
            self._recording_map[action_name].__exit__(None, None, None)
            del self._recording_map[action_name]

        if not _KINETO_AVAILABLE or self._emit_nvtx:
            return

        if self.profiler is not None and any(action_name.endswith(func) for func in self.STEP_FUNCTIONS):
            if self._schedule is not None:  # type: ignore
                self._schedule.pre_step(action_name)  # type: ignore

            # the default schedule requires a minimum of 5 steps to properly work: `wait=1, warmup=1, active=3`.
            # otherwise, this will raise a `segmentation fault`.
            if self._should_override_schedule():
                warning_cache.warn(
                    "The SDAAProfiler default schedule will be overridden as there is not enough "
                    "steps to properly record traces."
                )
                self._schedule = None
                self.profiler.schedule = torch.profiler.profiler._default_schedule_fn

            def on_trace_ready(profiler: _PROFILER) -> None:
                if self.dirpath is not None:
                    if self._export_to_chrome:
                        file_name = re.sub(r"[^a-zA-Z0-9]+", "_", action_name)
                        handler = tensorboard_trace_handler(
                            str(self.dirpath), self._prepare_filename(
                                action_name=file_name, extension="")
                        )
                        handler(profiler)

                    if self._export_to_flame_graph:
                        path = os.path.join(
                            self.dirpath, self._prepare_filename(
                                action_name=action_name, extension=".stack")
                        )
                        profiler.export_stacks(path, metric=self._metric)
                else:
                    rank_zero_warn(
                        "The SDAAProfiler failed to export trace as `dirpath` is None")

            if not self._has_on_trace_ready:
                self.profiler.on_trace_ready = on_trace_ready

            if self._schedule is not None:
                self.profiler.step_num = self._schedule.num_step
            self.profiler.step()
            self.profiler.add_metadata("Framework", "pytorch-lightning")

    def summary(self) -> str:
        return "Summary not supported for SDAA Profiler"
